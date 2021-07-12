"""
Listener ABC.
"""

import threading
import logging

from abc import ABC, abstractmethod
from collections import namedtuple
from typing import List, Union, Tuple

import numpy as np
from obspy import UTCDateTime
from obspy.core.event import Event, Origin
from obspy.core.event.origin import Pick


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

    threads = []
    client = None
    # TODO: old_events should be a property with getter and setter methods that are threadsafe!
    _old_events = []  # List of tuples of (event_id, event_time)
    keep = 86400  # Time in seconds to keep old events
    lock = threading.Lock()  # Lock for access to old_events

    @abstractmethod
    def run(self, *args, **kwargs):
        """ Run the listener """

    def get_old_events(self) -> List[EventInfo]:
        """ Threadsafe access to underlying list of tuples of old-events. """
        with self.lock:
            old_events = self._old_events
        return old_events

    def set_old_events(self, events: List[EventInfo]):
        with self.lock:
            self._old_events = events

    old_events = property(fget=get_old_events, fset=set_old_events)

    def remove_old_event(self, event: EventInfo):
        with self.lock:  # Make threadsafe
            self._old_events.remove(event)

    def extend(self, events: Union[EventInfo, List[EventInfo]]):
        """ Threadsafe way to add events to the cache """
        if isinstance(events, EventInfo):
            events = [events]
        with self.lock:
            self._old_events.extend(events)

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

    def background_run(self, *args, **kwargs):
        self.busy = True
        listening_thread = threading.Thread(
            target=self.run, args=args, kwargs=kwargs,
            name="ListeningThread")
        listening_thread.daemon = True
        listening_thread.start()
        self.threads.append(listening_thread)
        Logger.info("Started listening to {0}".format(self.client.base_url))

    def background_stop(self):
        self.busy = False
        for thread in self.threads:
            thread.join(timeout=10)
            if thread.is_alive():
                # Didn't join within timeout...
                thread.join()


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
