"""
Plotting for real-time seismic data.
"""
import numpy as np
import os
import chime
import logging
import threading
import datetime as dt
import asyncio

from typing import List, Iterable

from pyproj import Proj, Transformer

from bokeh.document import Document
from bokeh.plotting import figure
from bokeh.models import ColumnDataSource, HoverTool, Legend, WMTSTileSource
from bokeh.models.glyphs import MultiLine
from bokeh.models.formatters import DatetimeTickFormatter
from bokeh.layouts import gridplot, column
from bokeh.server.server import Server
from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler

from functools import partial

from obspy import Inventory

from earthquake_viewer.config.config import Config
from earthquake_viewer.listener.listener import EventInfo, PickInfo


STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")

Logger = logging.getLogger(__name__)

LOCALTZ = dt.datetime.now().astimezone().tzinfo

MOST_RECENT = 10  # Number of events to put in the event table
# Column titles keyed by event attribute
COLUMNS = {
    "time": "Origin-Time", 
    "latitude": "Latitude", 
    "longitude": "Longitude", 
    "depth": "Depth (km)", 
    "magnitude": "Magnitude"}
COLUMN_WIDTHS = [0.3, 0.175, 0.175, 0.175, 0.175]
NONEVENT = EventInfo(None, None, None, None, None, None, 
    p_picks=[PickInfo(None, None, None, None)], 
    s_picks=[PickInfo(None, None, None, None)])

CHIME = True  # Make a noise when new events are registered
chime.theme("zelda")

DPI = 100


class BokehPlotter(object):
    _timeout = 120  # Timeout limit for restarting streamer.

    def __init__(
        self,
        configuration: Config
    ):
        # Setup new event loop for plotting
        asyncio.set_event_loop(asyncio.new_event_loop())
        
        # start services to avoid delays
        self.config = configuration
        self.streamer = configuration.get_streamer()
        self.listener = configuration.get_listener()
        # Start background services
        if not self.listener.busy:
            Logger.info("Starting event listening service")
            self.listener.background_run(event_type="earthquake")
        if not self.streamer.streaming:
            Logger.info("Starting waveform streaming service")
            self.streamer.background_run()

        # Get the inventory for the map
        inv = configuration.get_inventory(level="station")

        self.hover = HoverTool(
            tooltips=[
                ("UTCDateTime", "@time{%m/%d %H:%M:%S}"),
                ("Amplitude", "@data")],
            formatters={'time': 'datetime'},
            mode='vline')
        self.map_hover = HoverTool(
            tooltips=[
                ("Latitude", "@lats"),
                ("Longitude", "@lons"),
                ("ID", "@id")])
        self.tools = "pan,wheel_zoom,reset"
        plot_width, plot_height = self.config.plotting.figure_size
        plot_width *= DPI
        plot_height *= DPI
        self.plot_options = {
            "plot_width": int(2 * (plot_width / 3)),
            "plot_height": int((plot_height - 20) / len(self.config.earthquake_viewer.seed_ids)),
            "tools": [self.hover], "x_axis_type": "datetime"}
        self.map_options = {
            "plot_width": int(plot_width / 3), "plot_height": plot_height,
            "tools": [self.map_hover, self.tools]}
        self.updateValue = True
        Logger.info("Initializing plotter")
        make_doc = partial(
            define_plot, 
            rt_client=self.streamer, 
            channels=self.config.earthquake_viewer.seed_ids,
            inventory=inv, 
            detections=self.listener.old_events,
            map_options=self.map_options,
            map_bounds=self.config.plotting.map_bounds,
            plot_options=self.plot_options, 
            plot_length=self.config.streaming.buffer_capacity,
            update_interval=self.config.plotting.update_interval, 
            offline=False, 
            lowcut=self.config.plotting.lowcut,
            highcut=self.config.plotting.highcut, 
            decimate=self.config.plotting.decimate,
            decay=self.config.plotting.event_history)

        self.apps = {'/eqv': Application(FunctionHandler(make_doc))}

        self.server = Server(self.apps)
        self.server.start()
        Logger.info("Plotting started")
        self.threads = []
    
    def show(self):
        """ run the plotting in a daemon thread. """
        plotting_thread = threading.Thread(
            target=self._bg_run, name="PlottingThread")
        plotting_thread.daemon = True
        plotting_thread.start()
        self.threads.append(plotting_thread)
        Logger.info("Started plotting")

    def _bg_run(self):
        print('Opening Bokeh application on http://localhost:5006/')
        self.server.io_loop.add_callback(self.server.show, "/")
        self.server.io_loop.start()

    def background_stop(self):
        """ Stop the background plotting thread. """
        self.server.io_loop.stop()
        for thread in self.threads:
            thread.join()


def define_plot(
    doc: Document,
    rt_client,
    channels: list,
    inventory: Inventory,
    detections: list,
    map_options: dict,
    map_bounds: tuple,
    plot_options: dict,
    plot_length: float,
    update_interval: int,
    data_color: str = "grey",
    lowcut: float = 1.0,
    highcut: float = 10.0,
    decimate: int = 1,
    offline: bool = False,
    decay: float = 86400,
):
    """
    Set up a bokeh plot for real-time plotting.

    Defines a moving data stream and a map.

    Parameters
    ----------
    doc
        Bokeh document to edit - usually called as a partial
    rt_client
        RealTimeClient streaming data
    channels
        Channels to plot
    inventory
        Inventory to plot
    detections
        Detections to plot - should be a list that is updated in place.
    map_options
        Dictionary of options for the map
    plot_options
        Dictionary of options for plotting in general
    plot_length
        Length of data plot
    update_interval
        Update frequency in seconds
    data_color
        Colour to data stream
    lowcut
        Lowcut for filtering data stream
    highcut
        Highcut for filtering data stream
    decimate
        Decimation factor.
    offline
        Flag to set time-stamps to data time-stamps if True, else timestamps
        will be real-time
    decay
        How long to retain decay event alphas over
    """
    # Set up the data source
    Logger.info("Getting stream to define plot")
    stream = rt_client.stream.copy().split().detrend()
    if lowcut and highcut:
        stream.filter("bandpass", freqmin=lowcut, freqmax=highcut)
        title = "Streaming data: {0}-{1} Hz bandpass".format(lowcut, highcut)
    elif lowcut:
        stream.filter("highpass", lowcut)
        title = "Streaming data: {0} Hz highpass".format(lowcut)
    elif highcut:
        stream.filter("lowpass", highcut)
        title = "Streaming data: {0} Hz lowpass".format(highcut)
    else:
        title = "Raw streaming data"
    if decimate and decimate > 1:
        stream.decimate(decimate)
    stream.merge()
    Logger.info(f"Have the stream: \n{stream}")

    station_lats, station_lons, station_ids = ([], [], [])
    for network in inventory:
        for station in network:
            station_lats.append(station.latitude)
            station_lons.append(station.longitude % 360)
            station_ids.append(station.code)

    # Get plot bounds in web mercator
    Logger.info("Defining map")
    transformer = Transformer.from_crs(
        "epsg:4326", "epsg:3857", always_xy=True)
    min_lon, max_lon, min_lat, max_lat = map_bounds
    Logger.info(f"Map bounds: {min_lon}, {min_lat} - {max_lon}, {max_lat}")
    bottom_left = transformer.transform(min_lon, min_lat)
    top_right = transformer.transform(max_lon, max_lat)
    map_x_range = (bottom_left[0], top_right[0])
    map_y_range = (bottom_left[1], top_right[1])

    station_x, station_y = ([], [])
    for lon, lat in zip(station_lons, station_lats):
        _x, _y = transformer.transform(lon, lat)
        station_x.append(_x)
        station_y.append(_y)

    # Empty data source for earthquakes
    detection_source = ColumnDataSource({
        'y': [], 'x': [], 'lats': [], 'lons': [], 'alphas': [],
        'sizes':[], 'id': []})
    station_source = ColumnDataSource({
        'y': station_y, 'x': station_x,
        'lats': station_lats, 'lons': station_lons, 'id': station_ids})

    Logger.info("Allocated data sources")
    trace_sources = {}
    trace_data_range = {}
    # Allocate empty arrays
    for channel in channels:
        try:
            tr = stream.select(id=channel)[0]
        except IndexError:
            Logger.warning(f"No data for {channel}")
            tr = None
        if tr:
            times = np.arange(
                tr.stats.starttime.datetime,
                (tr.stats.endtime + tr.stats.delta).datetime,
                step=dt.timedelta(seconds=tr.stats.delta))
            data = tr.data
        else:
            times, data = (
                np.array([], dtype=np.datetime64), np.array([], dtype=float))
        trace_sources.update(
            {channel: ColumnDataSource({'time': times, 'data': data})})
        if tr:
            trace_data_range.update({channel: (data.min(), data.max())})

    # Set up the map to go on the left side
    Logger.info("Adding features to map")
    map_plot = figure(
        title="Earthquake map", x_range=map_x_range, y_range=map_y_range,
        x_axis_type="mercator", y_axis_type="mercator", **map_options)
    url = 'http://a.basemaps.cartocdn.com/rastertiles/voyager/{Z}/{X}/{Y}.png'
    attribution = "Tiles by Carto, under CC BY 3.0. Data by OSM, under ODbL"
    map_plot.add_tile(WMTSTileSource(url=url, attribution=attribution))
    map_plot.circle(
        x="x", y="y", source=detection_source, fill_color="firebrick",
        line_color="grey", line_alpha=.2,
        fill_alpha="alphas", size="sizes")
    map_plot.triangle(
        x="x", y="y", size=10, source=station_source, color="blue", alpha=1.0)

    # Set up the trace plots
    Logger.info("Setting up streaming plot")
    trace_plots = []
    if not offline:
        now = dt.datetime.utcnow()
    else:
        now = max([tr.stats.endtime for tr in stream]).datetime
    p1 = figure(
        y_axis_location="right", title=title,
        x_range=[now - dt.timedelta(seconds=plot_length), now],
        plot_height=int(plot_options["plot_height"] * 1.2),
        **{key: value for key, value in plot_options.items()
           if key != "plot_height"})
    p1.yaxis.axis_label = None
    p1.xaxis.axis_label = None
    p1.min_border_bottom = 0
    p1.min_border_top = 0
    if len(channels) != 1:
        p1.xaxis.major_label_text_font_size = '0pt'
    p1_line = p1.line(
        x="time", y='data', source=trace_sources[channels[0]],
        color=data_color, line_width=1)
    legend = Legend(items=[(channels[0], [p1_line])])
    p1.add_layout(legend, 'right')

    datetick_formatter = DatetimeTickFormatter(
        days=["%m/%d"], months=["%m/%d"],
        hours=["%m/%d %H:%M:%S"], minutes=["%m/%d %H:%M:%S"],
        seconds=["%m/%d %H:%M:%S"], hourmin=["%m/%d %H:%M:%S"],
        minsec=["%m/%d %H:%M:%S"])
    p1.xaxis.formatter = datetick_formatter

    # Add detection lines
    Logger.info("Adding detection artists")
    oldest_time = now - dt.timedelta(seconds=plot_length)
    p_pick_source = _get_pick_times(detections, channels[0], phase="P",
                                    oldest_time=oldest_time)
    s_pick_source = _get_pick_times(detections, channels[0], phase="S",
                                    oldest_time=oldest_time)
    try:
        tr = stream.select(id=channels[0])[0]
    except IndexError:
        tr = None
    if tr:
        data_range = [int(min(tr.data)), int(max(tr.data))]
    else:
        data_range = [-1, 1]
    p_pick_source.update(
        {"pick_values": [data_range for _ in p_pick_source['picks']]})
    s_pick_source.update(
        {"pick_values": [data_range for _ in s_pick_source['picks']]})
    p_pick_sources = {channels[0]: ColumnDataSource(p_pick_source)}
    s_pick_sources = {channels[0]: ColumnDataSource(s_pick_source)}
    p_pick_lines = MultiLine(
        xs="picks", ys="pick_values", line_color="red", line_dash="dashed",
        line_width=1)
    s_pick_lines = MultiLine(
        xs="picks", ys="pick_values", line_color="blue", line_dash="dashed",
        line_width=1)
    p1.add_glyph(p_pick_sources[channels[0]], p_pick_lines)
    p1.add_glyph(s_pick_sources[channels[0]], s_pick_lines)

    trace_plots.append(p1)

    if len(channels) > 1:
        for i, channel in enumerate(channels[1:]):
            p = figure(
                x_range=p1.x_range,
                y_axis_location="right", **plot_options)
            p.yaxis.axis_label = None
            p.xaxis.axis_label = None
            p.min_border_bottom = 0
            # p.min_border_top = 0
            p_line = p.line(
                x="time", y="data", source=trace_sources[channel],
                color=data_color, line_width=1)
            legend = Legend(items=[(channel, [p_line])])
            p.add_layout(legend, 'right')
            p.xaxis.formatter = datetick_formatter

            # Add detection lines
            p_pick_source = _get_pick_times(detections, channel, phase="P",
                                            oldest_time=oldest_time)
            s_pick_source = _get_pick_times(detections, channel, phase="S",
                                            oldest_time=oldest_time)
            try:
                tr = stream.select(id=channel)[0]
            except IndexError:
                tr = None
            if tr:
                data_range = [int(min(tr.data)), int(max(tr.data))]
            else:
                data_range = [-1, 1]
            p_pick_source.update(
                {"pick_values": [data_range for _ in p_pick_source['picks']]})
            s_pick_source.update(
                {"pick_values": [data_range for _ in s_pick_source['picks']]})
            p_pick_sources.update({channel: ColumnDataSource(p_pick_source)})
            s_pick_sources.update({channel: ColumnDataSource(s_pick_source)})
            p_pick_lines = MultiLine(
                xs="picks", ys="pick_values", line_color="red", line_dash="dashed",
                line_width=1)
            s_pick_lines = MultiLine(
                xs="picks", ys="pick_values", line_color="blue", line_dash="dashed",
                line_width=1)
            p.add_glyph(p_pick_sources[channel], p_pick_lines)
            p.add_glyph(s_pick_sources[channel], s_pick_lines)

            trace_plots.append(p)

            if i != len(channels) - 2:
                p.xaxis.major_label_text_font_size = '0pt'
    plots = gridplot([[map_plot, column(trace_plots)]])

    previous_timestamps = {channel: None for channel in channels}
    for channel in channels:
        tr = stream.select(id=channel)
        if len(tr):
            previous_timestamps.update({channel: tr[0].stats.endtime})
    
    def update():
        Logger.info("Plot updating")
        _stream = rt_client.stream.split().detrend()
        if lowcut and highcut:
            _stream.filter("bandpass", freqmin=lowcut, freqmax=highcut)
        elif lowcut:
            _stream.filter("highpass", lowcut)
        elif highcut:
            _stream.filter("lowpass", highcut)
        _stream.merge()

        for _i, _channel in enumerate(channels):
            try:
                _tr = _stream.select(id=_channel)[0]
            except IndexError:
                Logger.debug("No channel for {0}".format(_channel))
                continue
            _prev = previous_timestamps[_channel] or _tr.stats.starttime
            new_samples = int(_tr.stats.sampling_rate * (
                    _prev - _tr.stats.endtime))
            if new_samples == 0:
                Logger.debug("No new data for {0}".format(_channel))
                continue
            _new_data = _tr.slice(starttime=_prev)
            new_times = np.arange(
                _new_data.stats.starttime.datetime,
                (_tr.stats.endtime + _tr.stats.delta).datetime,
                step=dt.timedelta(seconds=_tr.stats.delta))
            new_data = {'time': new_times[1:], 'data': _new_data.data[1:]}
            Logger.debug("Channel: {0}\tNew times: {1}\t New data: {2}".format(
                _tr.id, new_data["time"].shape, new_data["data"].shape))
            trace_sources[_channel].stream(
                new_data=new_data,
                rollover=int(plot_length * _tr.stats.sampling_rate))
            
            for phase, phase_source in zip(["P", "S"], [p_pick_sources, s_pick_sources]):
                new_picks = _get_pick_times(
                    detections, _channel, phase=phase, 
                    oldest_time=(_tr.stats.endtime - plot_length).datetime)
                new_picks.update({
                    'pick_values': [
                        [int(np.nan_to_num(
                            trace_sources[_channel].data['data']).max() * .9),
                        int(np.nan_to_num(
                            trace_sources[_channel].data['data']).min() * .9)]
                        for _ in new_picks['picks']]})
                phase_source[_channel].data = new_picks
            previous_timestamps.update({_channel: _tr.stats.endtime})
            Logger.debug("New data plotted for {0}".format(_channel))

        try:
            now = max([tr.stats.endtime for tr in _stream]).datetime
        except ValueError:
            return
        trace_plots[0].x_range.start = now - dt.timedelta(seconds=plot_length)
        trace_plots[0].x_range.end = now

        _update_template_alphas(
            detections, decay=decay, now=now,
            datastream=detection_source)
    Logger.info("Adding callback")
    doc.add_periodic_callback(update, update_interval)
    doc.title = "Earthquake Viewer"
    doc.add_root(plots)
    Logger.info("Plot defined")


def _update_template_alphas(
    detections: list,
    decay: float,
    now, 
    datastream
) -> None:
    """
    Update the template location datastream.

    Parameters
    ----------
    detections
        Detections to use to update the datastream
    decay
        Colour decay length in seconds
    now
        Reference time-stamp
    datastream
        Data stream to update
    """
    transformer = Transformer.from_crs(
        "epsg:4326", "epsg:3857", always_xy=True)
    lats, lons, alphas, ids, sizes = ([], [], [], [], [])
    _x, _y = ([], [])
    for detection in detections:
        lats.append(detection.latitude)
        lons.append(detection.longitude)

        ids.append(detection.event_id)
        x, y = transformer.transform(detection.longitude, detection.latitude)
        _x.append(x)
        _y.append(y)

        offset = (now - detection.time.datetime).total_seconds()
        alpha = 1. - (offset / decay)
        Logger.debug('Updating alpha to {0:.4f}'.format(alpha))
        alphas.append(alpha)
        sizes.append(detection.magnitude)
    sizes = _scale_mags(sizes)

    datastream.data = {
        'y': _y, 'x': _x, 'lats': lats, 'lons': lons,
        'alphas': alphas, 'id': ids, 'sizes': sizes}
    
    return


def _scale_mags(magnitudes: Iterable) -> Iterable:
    return [m ** 3 for m in magnitudes]


def _get_pick_times(
    detections: List[EventInfo],
    seed_id: str,
    phase: str,
    oldest_time: dt.datetime,
) -> dict:
    """
    Get new pick times from catalog for a given channel.

    Parameters
    ----------
    detections
        List of detections
    seed_id
        The full Seed-id (net.sta.loc.chan) for extract picks for
    ignore_channel
        Whether to return all picks for a given sensor (e.g. HH*)

    Returns
    -------
    Dictionary with one key ("picks") of the pick-times.
    """
    phase = phase.upper()
    assert phase in "PS"
    picks = []
    Logger.debug("Scanning {0} detections for new picks".format(
        len(detections)))
    net, sta, loc, chan = seed_id.split('.')
    for detection in detections:
        Logger.debug(f"Extracting picks from {detection}")
        if phase == "P":
            _picks = detection.p_picks
        else:
            _picks = detection.s_picks
        pick = [p for p in _picks if p.station == sta]
        for _pick in pick:
            if _pick.time > oldest_time:
                # Skip old picks!
                continue
            Logger.info("Plotting pick on {0} at {1}".format(
                seed_id, _pick.time))
            picks.append([_pick.time, _pick.time])
    return {"picks": picks}


if __name__ == "__main__":
    import doctest

    doctest.testmod()
