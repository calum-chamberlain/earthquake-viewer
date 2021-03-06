"""
Listener to an event stream.
"""
import time
import logging
import copy

from typing import Union, Callable
from queue import Empty

from obspy import UTCDateTime, Catalog
from obspy.core.event import Event

from earthquake_viewer.listener.listener import _Listener, summarise_event

Logger = logging.getLogger(__name__)


def filter_events(
    events: Union[list, Catalog],
    min_stations: int,
    auto_picks: bool,
    auto_event: bool,
    event_type: Union[list, str],
    **kwargs,
) -> list:
    """
    Filter events from a catalog based on some quality attributes.

    Parameters
    ----------
    events:
        Events to apply filter to.
    min_stations:
        Minimum number of stations for event to be kept.
    auto_picks:
        Whether to keep automatic picks or not.
    auto_event:
        Whether to keep automatic events or not.
    event_type:
        Event types to keep.

    Returns
    -------
    Events that pass the criteria
    """
    if isinstance(events, Catalog):
        events_out = copy.deepcopy(events.events)
    else:
        events_out = copy.deepcopy(events)

    _events_out = []
    for i, ev in enumerate(events_out):
        ev, keep = _qc_event(
            ev, min_stations=min_stations, auto_picks=auto_picks,
            auto_event=auto_event, event_type=event_type
        )
        if keep:
            _events_out.append(ev)
    return _events_out


def remove_unreferenced(catalog: Union[Catalog, Event]) -> Catalog:
    """ Remove un-referenced arrivals, amplitudes and station_magnitudes. """
    if isinstance(catalog, Event):
        catalog = Catalog([catalog])
    catalog_out = Catalog()
    for _event in catalog:
        event = _event.copy()
        pick_ids = {p.resource_id for p in event.picks}
        # Remove unreferenced arrivals
        for origin in event.origins:
            origin.arrivals = [
                arr for arr in origin.arrivals if arr.pick_id in pick_ids]
        # Remove unreferenced amplitudes
        event.amplitudes = [
            amp for amp in event.amplitudes if amp.pick_id in pick_ids]
        amplitude_ids = {a.resource_id for a in event.amplitudes}
        # Remove now unreferenced station magnitudes
        event.station_magnitudes = [
            sta_mag for sta_mag in event.station_magnitudes
            if sta_mag.amplitude_id in amplitude_ids]
        station_magnitude_ids = {
            sta_mag.resource_id for sta_mag in event.station_magnitudes}
        # Remove unreferenced station_magnitude_contributions
        for magnitude in event.magnitudes:
            magnitude.station_magnitude_contributions = [
                sta_mag_contrib
                for sta_mag_contrib in magnitude.station_magnitude_contributions
                if sta_mag_contrib.station_magnitude_id in station_magnitude_ids]
        catalog_out.append(event)

    return catalog_out


def _qc_event(
    event: Event,
    min_stations: int = None,
    auto_picks: bool = True,
    auto_event: bool = True,
    event_type: Union[list, str] = None,
) -> tuple:
    """
    QC an individual event - removes picks in place.

    Returns
    -------
    tuple of (event: Event, keep: bool)
    """
    if event_type is not None and isinstance(event_type, str):
        event_type = [event_type]
    if event_type is not None and event.event_type not in event_type:
        return event, False
    elif not auto_event:
        if "manual" not in [ori.evaluation_mode for ori in event.origins]:
            return event, False
    if not auto_picks:
        pick_ids_to_remove = [
            p.resource_id for p in event.picks
            if p.evaluation_mode == "automatic"
        ]
        # remove arrivals and amplitudes and station_magnitudes
        event.picks = [
            p for p in event.picks if p.resource_id not in pick_ids_to_remove
        ]
        event = remove_unreferenced(event)[0]
    stations = {p.waveform_id.station_code for p in event.picks}
    if len(stations) < min_stations:
        return event, False
    return event, True


class CatalogListener(_Listener):
    """
    Client query class for obspy clients with a `get_events` service.

    Parameters
    ----------
    client:
        Client to query - must have at least a `get_events` method.
    catalog:
        Catalog of past events - can be empty. Any new events will be compared
        to this catalog and only added to the template bank if they are not
        in the original catalog.
    catalog_lookup_kwargs:
        Dictionary of keyword arguments for `client.get_events`.
    interval:
        Interval for querying the client in seconds. Note that rapid queries
        may not be more efficient, and will almost certainly piss off your
        provider.
    keep:
        Time in seconds to keep events for in the catalog in memory. Will not
        remove old events on disk. Use to reduce memory consumption.
    refresh:
        Whether to refresh the catalog or just get new events - useful if the
        client may update the catalog.
    """
    busy = False
    _test_start_step = 0  # Number of seconds prior to `now` used for testing.
    _speed_up = 1  # Multiplier for query intervals, used for synthesising
                   # previous sequences, not general purpose.

    def __init__(
        self,
        client,
        catalog: Catalog = None,
        catalog_lookup_kwargs: dict = None,
        interval: float = 10,
        keep: float = 86400,
        refresh: bool = True,
    ):
        _Listener.__init__(self)  # Init ABC to get the queues
        self.client = client
        if catalog is None:
            catalog = Catalog()
        self.old_events = [summarise_event(ev) for ev in catalog]
        self.catalog_lookup_kwargs = catalog_lookup_kwargs or dict()
        self.interval = interval
        self.keep = keep
        self.threads = []
        self.triggered_events = Catalog()
        self.busy = False
        self.previous_time = UTCDateTime.now()
        self.refresh = refresh

    def __repr__(self):
        """
        ..rubric:: Example
        >>> from obspy.clients.fdsn import Client
        >>> listener = CatalogListener(
        ...     client=Client("GEONET"), catalog=Catalog(),
        ...     catalog_lookup_kwargs=dict(
        ...         latitude=-45, longitude=175, maxradius=2))
        >>> print(listener) # doctest: +NORMALIZE_WHITESPACE
        CatalogListener(client=Client(http://service.geonet.org.nz),\
        catalog=Catalog(0 events), interval=10, **kwargs)
        """
        print_str = (
            "CatalogListener(client=Client({0}), catalog=Catalog({1} events), "
            "interval={2}, **kwargs)".format(
                self.client.base_url, len(self.old_events), self.interval))
        return print_str

    @property
    def sleep_interval(self):
        return self.interval / self._speed_up

    def run(
        self,
        min_stations: int = 0,
        auto_event: bool = True,
        auto_picks: bool = True,
        event_type: Union[list, str] = None,
        filter_func: Callable = None,
        starttime: UTCDateTime = None,
        **filter_kwargs,
    ) -> None:
        """
        Run the listener. New events will be added to the template_bank.

        Parameters
        ----------
        min_stations:
            Minimum number of stations for an event to be added to the
            TemplateBank
        auto_event:
            If True, both automatic and manually reviewed events will be
            included. If False, only manually reviewed events will be included
        auto_picks:
            If True, both automatic and manual picks will be included. If False
            only manually reviewed picks will be included. Note that this is
            done **before** counting the number of stations.
        event_type
            List of event types to keep.
        filter_func
            Function used for filtering. If left as none, this will use the
            `catalog_listener.filter_events` function.
        starttime
            When to start to get events from, defaults to the last time
            the listener was run.
        filter_kwargs:
            If the `filter_func` has changed then this should be the
            additional kwargs for the user-defined filter_func.
        """
        if starttime is None:
            self.previous_time -= self._test_start_step
        else:
            self.previous_time = starttime
        self._stop_called = False  # Reset this - if someone called run,
        # they probably want us to run!

        loop_duration = 0  # Timer for loop, used in synthesising speed-ups
        while not self._stop_called:
            tic = time.time()  # Timer for loop, used in synthesising speed-ups
            if self._test_start_step > 0:
                # Still processing past data
                self._test_start_step -= loop_duration * self._speed_up
                self._test_start_step += loop_duration
                # Account for UTCDateTime.now() already including loop_
                # duration once.
            elif self._test_start_step < 0:
                # We have gone into the future!
                raise NotImplementedError(
                    "Trying to access future data: spoilers not allowed")
            now = UTCDateTime.now() - self._test_start_step
            # Remove old events from cache
            self._remove_old_events(now)
            # Check for new events - add in a pad of five times the
            # checking interval to ensure that slow-to-update events are
            # included.
            if self.refresh:
                _starttime = now - self.keep
            else:
                _starttime = self.previous_time - (5 * self.interval)
            Logger.info("Checking for new events between {0} and {1}".format(
                _starttime, now))
            try:
                new_events = self.client.get_events(
                    starttime=_starttime, endtime=now,
                    **self.catalog_lookup_kwargs)
            except Exception as e:
                if "No data available for request" in e.args[0]:
                    Logger.debug("No new data")
                else:  # pragma: no cover
                    Logger.error(
                        "Could not download data between {0} and {1}".format(
                            _starttime, now))
                    Logger.error(e)
                time.sleep(self.sleep_interval)
                toc = time.time()  # Timer for loop, used in synthesising speed-ups
                loop_duration = toc - tic
                continue
            if new_events is not None and len(new_events) > 0:
                if filter_func is not None:
                    new_events = filter_func(
                        new_events, min_stations=min_stations,
                        auto_picks=auto_picks, auto_event=auto_event,
                        event_type=event_type, **filter_kwargs)
                if not self.refresh:
                    old_event_ids = [e.event_id for e in self.old_events]
                    new_events = Catalog(
                        [ev for ev in new_events if ev.resource_id
                         not in old_event_ids])
                Logger.info("{0} new events between {1} and {2}".format(
                    len(new_events), _starttime, now))
                if len(new_events) > 0:
                    Logger.info("Adding {0} new events to the database".format(
                        len(new_events)))
                    for event in new_events:
                        try:
                            origin = (
                                event.preferred_origin() or event.origins[0])
                        except IndexError:
                            continue
                        try:
                            magnitude = (
                                event.preferred_magnitude() or
                                event.magnitudes[0])
                        except IndexError:
                            continue
                        Logger.info(
                            "Event {0}: M {1:.1f}, lat: {2:.2f}, "
                            "long: {3:.2f}, depth: {4:.2f}km".format(
                                event.resource_id.id, magnitude.mag,
                                origin.latitude, origin.longitude,
                                origin.depth / 1000.))
                    event_info = [summarise_event(ev) for ev in new_events]
                    if not self.refresh:
                        self.extend(event_info)
                    else:
                        # Update old events
                        old_event_dict = {
                            e.event_id: e for e in self.old_events}
                        for event in event_info:
                            if event.event_id in old_event_dict.keys():
                                Logger.debug(f"Updating event {event.event_id}")
                                # Remove the previous version and update it
                                old_event = old_event_dict.pop(event.event_id)
                                self.remove_old_event(old_event)
                            self.extend(event)
                            old_event_dict.update({event.event_id: event})
                    Logger.debug("Old events current state: {0}".format(
                        self.old_events))
            self.previous_time = now
            # Sleep in steps to make death responsive
            _sleep_step = min(0.5, self.sleep_interval)
            _slept = 0.0
            while _slept < self.sleep_interval:
                _tic = time.time()
                time.sleep(_sleep_step)  # Need to sleep to allow other threads to run
                # Murder check
                try:
                    kill = self._killer_queue.get(block=False)
                except Empty:
                    kill = False
                if kill:
                    Logger.warning(
                        "Run termination called - poison received.")
                    self._stop_called = True
                    break
                _toc = time.time()
                _slept += _toc - _tic
            toc = time.time()  # Timer for loop, used in synthesising speed-ups
            loop_duration = toc - tic
        self.busy = False
        return


if __name__ == "__main__":
    import doctest

    doctest.testmod()
