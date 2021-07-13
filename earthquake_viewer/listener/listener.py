"""
Listener ABC.
"""

import logging
import time

from abc import ABC, abstractmethod
from collections import namedtuple
from typing import List, Union, Tuple
from queue import Empty, Full

import numpy as np
from obspy import UTCDateTime
from obspy.core.event import Event, Origin
from obspy.core.event.origin import Pick

import platform
if platform.system() != "Linux":
    warnings.warn("Currently Process-based streaming is only supported on "
                  "Linux, defaulting to Thread-based streaming - you may run "
                  "into delayed plotting when updating often")
    import threading as multiprocessing
    from queue import Queue
    from threading import Thread as Process
else:
    import multiprocessing
    from multiprocessing import Queue, Process

Logger = logging.getLogger(__name__)


EventInfo = namedtuple(
    "EventInfo", ("event_id", "time", "latitude", "longitude", "depth",
                  "magnitude", "p_picks", "s_picks"))

PickInfo = namedtuple(
    "PickInfo", ("time", "seed_id", "phase_hint", "station"))


class _Listener(ABC):
    """
    Abstract base class for listener objects - anything to be used by the
    Reactor should fit in this scope.
    """
    busy = False
    started = False

    processes = []
    client = None
    keep = 86400  # Time in seconds to keep old events
    lock = multiprocessing.Lock()  # Lock for access to old_events
    _stop_called = False

    def __init__(self) -> None:
        # Queues for comms
        self._catalog_in_queue = Queue()
        self._old_events_queue = Queue(maxsize=1)

        # Poison!
        self._killer_queue = Queue(maxsize=1)
        self._dead_queue = Queue(maxsize=1)

        self.__old_events = []  # List of EventInfo namedtuples

    @abstractmethod
    def run(self, *args, **kwargs):
        """ Run the listener """

    @property
    def old_events(self) -> List[EventInfo]:
        try:
            self.__old_events = self._old_events_queue.get(block=False)
            # Need to put it back for future processes
            try:
                self._old_events_queue.put(self.__old_events, block=False)
            except Full:
                pass
        except Empty:
            Logger.debug("No events in queue")
            pass
        return self.__old_events

    @old_events.setter
    def old_events(self, events: List[EventInfo]):
        try:
            self._old_events_queue.put(events, block=False)
            Logger.debug("Put events into queue")
        except Full:
            # Empty it
            try:
                self._old_events_queue.get(block=False)
            except Empty:
                pass
            try:
                self._old_events_queue.put(events, timeout=10)
            except Full:
                Logger.error("Could not update old events - queue is full")

    def remove_old_event(self, event: EventInfo):
        old_events = self.old_events
        try:
            old_events.remove(event)
        except ValueError:
            Logger.warning(f"{event} not in old_events")
            return
        # Update old events
        self.old_events = old_events

    def extend(self, events: Union[EventInfo, List[EventInfo]]):
        """ Threadsafe way to add events to the cache """
        if isinstance(events, EventInfo):
            events = [events]
        old_events = self.old_events
        old_events.extend(events)
        self.old_events = old_events

    def _remove_old_events(self, endtime: UTCDateTime) -> None:
        """
        Expire old events from the cache.

        Parameters
        ----------
        endtime
            The time to calculate time-difference relative to. Any events
            older than endtime - self.keep will be removed.
        """
        if len(self.old_events) == 0:
            return
        time_diffs = np.array([endtime - tup[1] for tup in self.old_events])
        filt = time_diffs <= self.keep
        # Need to remove in-place, without creating a new list
        for i, old_event in enumerate(list(self.old_events)):
            if not filt[i]:
                self.remove_old_event(old_event)

    def _clear_killer(self):
        """ Clear the killer Queue. """
        while True:
            try:
                self._killer_queue.get(block=False)
            except Empty:
                break
        while True:
            try:
                self._dead_queue.get(block=False)
            except Empty:
                break

    def _bg_run(self, *args, **kwargs):
        while self.busy:
            self.run(*args, **kwargs)
        Logger.info("Running stopped, busy set to false")
        try:
            self._dead_queue.get(block=False)
        except Empty:
            pass
        self._dead_queue.put(True)
        return

    def background_run(self, *args, **kwargs):
        self.busy = True
        self._clear_killer()
        listening_thread = Process(
            target=self._bg_run, args=args, kwargs=kwargs,
            name="ListeningProcess")
        # listening_thread.daemon = True
        listening_thread.start()
        self.processes.append(listening_thread)
        Logger.info("Started listening to {0}".format(self.client.base_url))

    def background_stop(self):
        Logger.info("Adding Poison to Kill Queue")
        # Run communications before termination
        old_events = self.old_events

        self._killer_queue.put(True)
        Logger.debug("Adding old events to local buffer")
        self.old_events = old_events

        Logger.debug("Waiting for internal process to stop")
        while self.busy:
            try:
                self.busy = not self._dead_queue.get(block=False)
            except Empty:
                time.sleep(1)
                pass
        Logger.debug("Listener stopped")

        for queue in [self._catalog_in_queue, self._old_events_queue,
                      self._killer_queue, self._dead_queue]:
            while True:
                try:
                    queue.get(block=False)
                except Empty:
                    break
        # join the processes
        for process in self.processes:
            Logger.info("Joining process")
            process.join(5)
            if hasattr(process, 'exitcode') and process.exitcode:
                Logger.info("Process failed to join, terminating")
                process.terminate()
                Logger.info("Terminated")
                process.join()
            Logger.info("Process joined")
        self.processes = []
        self.busy = False


def event_origin(event: Event) -> Origin:
    try:
        origin = event.preferred_origin() or event.origins[-1]
    except IndexError:
        origin = None
    return origin


def event_time(event: Event) -> UTCDateTime:
    """
    Get the origin or first pick time of an event.

    Parameters
    ----------
    event:
        Event to get a time for

    Returns
    -------
    Reference time for event.
    """
    origin = event_origin(event)
    if origin is not None:
        return origin.time
    if len(event.picks) == 0:
        return UTCDateTime(0)
    return min([p.time for p in event.picks])


def event_magnitude(event: Event) -> float:
    """ Get a magnitude for an event - returns None if none set. """
    try:
        mag = event.preferred_magnitude() or event.magnitudes[-1]
    except IndexError:
        return None
    try:
        mag = mag.mag
    except AttributeError:
        return None
    return mag


def event_latitude(event: Event) -> float:
    origin = event_origin(event)
    if origin is None:
        return None
    try:
        return origin.latitude
    except AttributeError:
        return None


def event_longitude(event: Event) -> float:
    origin = event_origin(event)
    if origin is None:
        return None
    try:
        return origin.longitude
    except AttributeError:
        return None


def event_depth(event: Event) -> float:
    origin = event_origin(event)
    if origin is None:
        return None
    try:
        return origin.depth
    except AttributeError:
        return None


def summarise_pick(pick: Pick) -> PickInfo:
    return PickInfo(time=pick.time.datetime, 
                    seed_id=pick.waveform_id.get_seed_string(),
                    station=pick.waveform_id.station_code,
                    phase_hint=pick.phase_hint)


def event_picks(event: Event) -> Tuple[List[PickInfo], List[PickInfo]]:
    p_picks = [summarise_pick(p) 
               for p in event.picks if p.phase_hint.lower().startswith("p")]
    s_picks = [summarise_pick(p) 
               for p in event.picks if p.phase_hint.lower().startswith("s")]
    return p_picks, s_picks


def summarise_event(event: Event) -> EventInfo:
    p_picks, s_picks = event_picks(event)
    return EventInfo(
        event.resource_id.id, event_time(event), event_latitude(event),
        event_longitude(event), event_depth(event), event_magnitude(event),
        p_picks, s_picks)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
