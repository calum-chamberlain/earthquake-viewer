"""
Animated near-real-time plotting of streaming data.
"""
import time

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import logging
import matplotlib
import os
import datetime as dt
import numpy as np
import copy
import chime


from typing import Iterable

from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.colorbar import ColorbarBase
import matplotlib.dates as mdates
import matplotlib.table as mpltable
from mpl_toolkits.axes_grid1 import make_axes_locatable

from cartopy.mpl.ticker import LongitudeLocator, LatitudeLocator

from obspy import UTCDateTime, Stream

from earthquake_viewer.config.config import Config
from earthquake_viewer.listener.listener import EventInfo, PickInfo

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "images")

DEPTH_CMAP = copy.copy(cm.get_cmap("plasma_r")) 
NORM = Normalize(vmin=0, vmax=300.0)
DEPTH_CMAP.set_over("lime")

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

PCOLOR = "red"
SCOLOR = "purple"


def _scale_mags(magnitudes: Iterable) -> Iterable:
    return [m ** 3 for m in magnitudes]


def _blit_draw(self, artists):
    # Handles blitted drawing, which renders only the artists given instead
    # of the entire figure.
    updated_ax = {a.axes for a in artists}
    # Enumerate artists to cache axes' backgrounds. We do not draw
    # artists yet to not cache foreground from plots with shared axes
    for ax in updated_ax:
        # If we haven't cached the background for the current view of this
        # axes object, do so now. This might not always be reliable, but
        # it's an attempt to automate the process.
        cur_view = ax._get_view()
        view, bg = self._blit_cache.get(ax, (object(), None))
        bbox = ax.get_tightbbox(ax.figure.canvas.get_renderer())
        if cur_view != view:
            self._blit_cache[ax] = (
                cur_view, ax.figure.canvas.copy_from_bbox(bbox))
    # Make a separate pass to draw foreground.
    for a in artists:
        a.axes.draw_artist(a)
    # After rendering all the needed artists, blit each axes individually.
    for ax in updated_ax:
        bbox = ax.get_tightbbox(ax.figure.canvas.get_renderer())
        ax.figure.canvas.blit(bbox)

# MONKEY PATCH!!
matplotlib.animation.Animation._blit_draw = _blit_draw


def _time_string(seconds: int) -> str:
    """
    Get a string best fiting the elapsed time.

    Parameters
    ----------
    seconds:
        Total seconds to represent
    """
    value, unit = seconds, "seconds"
    if 3600 > value >= 60:
        value /= 60
        unit = "minutes"
    elif 86400 > value >= 3600:
        value /= 3600
        unit = "hours"
    elif value >= 86400:
        value /= 86400
        unit = "days"

    return f"{value:.1f} {unit}"


class Plotter(object):
    def __init__(
        self,
        configuration: Config,
    ):
        matplotlib.rcParams['toolbar'] = 'None'
        try:
            plt.style.use(configuration.plotting.style)
        except:
            plt.style.use(
                os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             configuration.plotting.style))
        fig = plt.figure(figsize=configuration.plotting.figure_size)
        if configuration.plotting.plot_map:
            # Factorise the width to get a sensible number of columns
            largest_factor = 100 % configuration.plotting.map_width_percent
            ncols = 100 // largest_factor
            map_width = (
                configuration.plotting.map_width_percent // largest_factor)
            # Set up gs for this
            gs = fig.add_gridspec(
                nrows=10,  # configuration.earthquake_viewer.n_chans,
                ncols=ncols)
            self.map_ax = fig.add_subplot(
                gs[0:8, 0:map_width],
                projection=configuration.plotting.map_projection)
            for spine in self.map_ax.spines.values():
                spine.set_edgecolor('black')
            divider = make_axes_locatable(self.map_ax)
            self.cbar_ax = divider.append_axes(
                position="bottom", size='5%', pad=0.05, axes_class=plt.Axes)
            self.table_ax = fig.add_subplot(gs[8:, 0:map_width])
            self.table_ax.set_axis_off()
            self.event_table = mpltable.table(
                ax=self.table_ax, loc="center",
                colLabels=list(COLUMNS.values()),
                colWidths=COLUMN_WIDTHS,
                cellText=[[None for _ in COLUMNS] 
                          for _ in range(MOST_RECENT)])
            self._events_in_table = set()
            self.event_table.auto_set_font_size(False)
            self.event_table.set_fontsize(8)
            # Get the locations of stations
            inv = configuration.get_inventory(level="station")
            self._station_locations = dict()
            for network in inv:
                for station in network:
                    self._station_locations.update(
                        {f"{network.code}.{station.code}":
                         (station.longitude, station.latitude)})
        else:
            Logger.info("Map plotting disabled")
            self.map_ax, self._station_locations = None, None
            map_width = 0
            gs = fig.add_gridspec(nrows=1, ncols=1)
        self.waveform_axes = fig.add_subplot(gs[:, map_width:])
        self.waveform_axes.yaxis.tick_right()
        self.waveform_axes.yaxis.set_label_position("right")
        ticks, labels = zip(*[(i, seed_id) for i, seed_id in enumerate(
            reversed(configuration.earthquake_viewer.seed_ids))])
        self.waveform_axes.set_yticks(ticks)
        self.waveform_axes.set_yticklabels(labels)
        self._data_offsets = {label: tick for tick, label in zip(ticks, labels)}
        fig.subplots_adjust(hspace=0)
        # Define properties
        self.fig = fig
        self.config = configuration
        self.streamer = configuration.get_streamer()
        self.listener = configuration.get_listener()
        self._previous_map_update = (
                UTCDateTime.now() - self.config.plotting.event_history)
        self._plotted_event_ids = list()
        self._previous_plot_time = {
            key: UTCDateTime(1970, 1, 1)
            for key in self.config.earthquake_viewer.seed_ids}
        self._last_data = UTCDateTime(0)
        # Define artists
        self.map_scatters = None
        self.waveform_lines = {
            key: None for key in self.config.earthquake_viewer.seed_ids}

        # Define pick times plotted
        self.p_times = {
            key: [] for key in self.config.earthquake_viewer.seed_ids}
        self.s_times = {
            key: [] for key in self.config.earthquake_viewer.seed_ids}

        # Add legend showing pick labels
        _p_line, = self.waveform_axes.plot([], [], c=PCOLOR)
        _s_line, = self.waveform_axes.plot([], [], c=SCOLOR)
        self.waveform_axes.legend(
            title="Phase picks", handles=[_p_line, _s_line],
            labels=["P", "S"], frameon=True, fancybox=True, loc="upper left")

        # Add logos and info
        # VUW logo
        vuw_logo_ax = fig.add_axes(
            [0.9, 0.9, 0.1, 0.1], anchor='NE', zorder=-1)
        vuw_logo = plt.imread(os.path.join(STATIC_DIR, "vuw_logo.png"))
        vuw_logo_ax.imshow(vuw_logo)
        vuw_logo_ax.text(
            x=0.5, y=-0.4, s="Code by Calum Chamberlain", 
            transform=vuw_logo_ax.transAxes, ha="center", va="bottom")
        vuw_logo_ax.axis('off')

        # GeoNet logo
        data_logo_ax = fig.add_axes(
            [0.0, 0.9, 0.1, 0.1], anchor='NW', zorder=-1)
        data_logo = plt.imread(
            os.path.join(STATIC_DIR, "Data_source_logo.jpg"))
        data_logo_ax.imshow(data_logo)
        data_logo_ax.text(
            x=0.5, y=-0.4, s=f"Data from {configuration.plotting.map_client}", 
            transform=data_logo_ax.transAxes, ha="center", va="bottom")
        data_logo_ax.axis('off')

        # Start background services
        if not self.listener.busy:
            Logger.info("Starting event listening service")
            self.listener.background_run(event_type="earthquake")
        if not self.streamer.streaming:
            Logger.info("Starting waveform streaming service")
            self.streamer.background_run()
        return

    def initialise_plot(self):
        if self.map_ax:
            self._initialise_map()
        # Initialise empty waveforms
        for i, seed_id in enumerate(reversed(self.config.earthquake_viewer.seed_ids)):
            Logger.info(f"Initialising for {seed_id}")
            line = Line2D([0], [i], linewidth=0.5)  #, color="k")
            self.waveform_axes.add_line(line)
            self.waveform_lines.update({seed_id: line})
        self.waveform_axes.set_ylim(
            -0.5, len(self.config.earthquake_viewer.seed_ids) -.5)
        self.waveform_axes.grid(True)
        tickformat = mdates.DateFormatter("%H:%M", tz=LOCALTZ)
        self.waveform_axes.xaxis.set_major_formatter(tickformat)
        self.waveform_axes.set_title(
            f"Data filtered between {self.config.plotting.lowcut} and"
            f" {self.config.plotting.highcut} Hz")
        return self.fig

    def _initialise_map(self):
        # Make the blank map!
        if self.config.plotting.global_map:
            self.map_ax.set_global()
            self.map_ax.stock_img()  # Plot a nice image on the globe
        else:
            Logger.info(
                f"Setting map extent to {self.config.plotting.map_bounds}")
            self.map_ax.set_extent(self.config.plotting.map_bounds,
                                   crs=ccrs.PlateCarree())
            self.map_ax.set_facecolor("white")  # oceans
            if self.config.plotting.latitude_range < 3:
                resolution = "h"
            elif self.config.plotting.latitude_range < 15:
                resolution = "i"
            else:
                resolution = "l"
            coast = cfeature.GSHHSFeature(
                scale=resolution, levels=[1], facecolor="lightgrey",
                edgecolor=None, alpha=1.0)
            self.map_ax.add_feature(coast)
        gl = self.map_ax.gridlines(
            draw_labels=True, color="black", alpha=0.5, zorder=2)
        gl.xlocator = LongitudeLocator(
            nbins=8, steps=[1, 2, 3, 4, 5, 6, 10], prune=None, min_n_ticks=5)
        gl.ylocator = LatitudeLocator(
            nbins=5, steps=[1, 2, 4, 5, 10], prune="both", min_n_ticks=3)
        gl.right_labels = False
        gl.bottom_labels = False
        if self._station_locations is not None:
            self.map_ax.scatter(
                [val[0] for val in self._station_locations.values()],
                [val[1] for val in self._station_locations.values()],
                marker="^", color="red", zorder=100,
                transform=ccrs.PlateCarree())
            if self.config.plotting.label_stations:
                for station_name, location in self._station_locations.items():
                    station_name = station_name.split('.')[-1]
                    self.map_ax.text(
                        location[0], location[1], station_name,
                        horizontalalignment='right',
                        verticalalignment="bottom", 
                        transform=ccrs.PlateCarree(),
                        bbox=dict(boxstyle='round', facecolor='white', 
                                  alpha=0.5))
        # Make an empty scatter artist
        self.map_scatters = self.map_ax.scatter(
            [], [], s=[], facecolors=[], edgecolors="k",
            transform=ccrs.PlateCarree(), cmap=DEPTH_CMAP)
        # Make a most recent scatter
        self.most_recent_scatter = self.map_ax.scatter(
            [], [], s=[], facecolors=[], edgecolors="gold", marker="*",
            transform=ccrs.PlateCarree(), cmap=DEPTH_CMAP)
        # Magnitude scale
        mag_scale_range = np.array([1.0, 3.0, 5.0, 7.0])
        mag_mapper = [self.map_ax.plot(
            [], [], 'ok', markersize=(m ** 3) ** .5, 
            transform=ccrs.PlateCarree())[0] 
                      for m in mag_scale_range]
        mag_labels = [f"{m:.1f}" for m in mag_scale_range]
        self.map_ax.legend(
            title="Magnitude", handles=mag_mapper, labels=mag_labels,
            frameon=True, fancybox=True)

        # Colorbar
        cb = ColorbarBase(
            self.cbar_ax, norm=NORM, cmap=DEPTH_CMAP, orientation='horizontal')
        cb.set_label("Depth (km)")
        if hasattr(cb, "update_ticks"):
            cb.update_ticks()

        self.map_ax.set_title(
            "Events from the last "
            f"{_time_string(self.config.plotting.event_history)}")
        return

    def update_map(self):
        """ Get new events and plot. """
        now = UTCDateTime.now()
        if len(self.listener.old_events) == 0:
            return []
        listener_events = self.listener.old_events
        # Plot the new events!
        positions = [(ev.longitude, ev.latitude) for ev in listener_events]
        depths = np.array([ev.depth for ev in listener_events]) / 1000.0
        mags = np.array([ev.magnitude for ev in listener_events])
        times = np.array([now - ev.time for ev in listener_events])
        alphas = 1 - (times / self.config.plotting.event_history)
        # Ensure boundedness
        alphas[alphas < 0] = 0
        alphas[alphas > 1] = 1
        colors = [DEPTH_CMAP(NORM(d), alpha=a) for a, d in zip(alphas, depths)]
        edgecolors = [(0, 0, 0, a) for a in alphas]
        # Update the content of the artist!
        self.map_scatters.set_offsets(positions)
        self.map_scatters.set_sizes(mags ** 3)
        self.map_scatters.set_facecolors(colors)
        self.map_scatters.set_edgecolors(edgecolors)

        # Update the most recent scatter
        most_recent_time = min(times)
        most_recent_index = times.argmin()
        self.most_recent_scatter.set_offsets([positions[most_recent_index]])
        self.most_recent_scatter.set_facecolors([colors[most_recent_index]])
        size = 300 - ((most_recent_time + 1) / 1800)
        self.most_recent_scatter.set_sizes([size])

        return [self.map_scatters, self.most_recent_scatter]

    def update_waveforms(self):
        # Get data from buffers
        now = UTCDateTime.now()
        plot_starttime = now - self.config.streaming.buffer_capacity
        if self._last_data >= self.streamer.last_data:
            Logger.info("No new data")
            self.waveform_axes.set_xlim(plot_starttime.datetime, now.datetime)
            return [self.waveform_axes]
        stream = self.streamer.stream.copy().merge()

        for tr in stream:
            seed_id = tr.id
            station = tr.stats.station
            plot_lim = self._previous_plot_time.get(seed_id, None)
            if plot_lim is None:
                Logger.debug(
                    f"{seed_id} not in {self._previous_plot_time.keys()}")
                continue
            Logger.debug(f"Working on data for {seed_id} ending {tr.stats.endtime} - last plotted {plot_lim}")
            if tr.stats.endtime <= plot_lim:
                continue  # No new data
            tic = time.perf_counter()
            if self.config.plotting.lowcut and self.config.plotting.highcut:
                tr = tr.split().detrend().filter(
                    "bandpass", freqmin=self.config.plotting.lowcut,
                    freqmax=self.config.plotting.highcut)
            elif self.config.plotting.lowcut:
                tr = tr.split().detrend().filter(
                    "highpass", self.config.plotting.lowcut)
            elif self.config.plotting.highcut:
                tr = tr.split().detrend().filter(
                    "lowpass", self.config.plotting.highcut)
            if self.config.plotting.decimate > 1:
                tr = tr.split().decimate(self.config.plotting.decimate)
            if isinstance(tr, Stream):
                tr = tr.merge()[0]
            toc = time.perf_counter()
            Logger.debug(f"\tProcessing for {seed_id} took {toc - tic:.3f}s")

            tic = time.perf_counter()
            self._previous_plot_time.update({seed_id: tr.stats.endtime})
            times = tr.times("matplotlib")
            data = tr.data.astype(float)

            # Normalize and offset
            data /= 2.5 * np.abs(data).max()
            data += self._data_offsets[seed_id]

            # Update!
            self.waveform_lines[seed_id].set_data(times, data)
            toc = time.perf_counter()
            Logger.debug(f"\tPlotting for {seed_id} took {toc - tic:.3f}s")

            # Get picks - only plot new picks!
            tic = time.perf_counter()
            p_picks = [p for ev in self.listener.old_events 
                       for p in ev.p_picks if p.station == station
                       and p.time > plot_starttime]
            s_picks = [p for ev in self.listener.old_events 
                       for p in ev.s_picks if p.station == station
                       and p.time > plot_starttime]
            for pcolor, _picks, _times in zip(
                    [PCOLOR, SCOLOR], [p_picks, s_picks],
                    [self.p_times, self.s_times]):
                for pick in _picks:
                    pick_time = mdates.date2num(pick.time)
                    if pick_time in _times[seed_id]:
                        continue
                    self.waveform_axes.vlines(
                        pick_time, ymin=self._data_offsets[seed_id] - .4,
                        ymax=self._data_offsets[seed_id] + .4, color=pcolor)
                    _times[seed_id].append(pick_time)
            toc = time.perf_counter()
            Logger.debug(f"\tPlotting picks for {seed_id} took {toc - tic:.3f}s")

        plot_starttime = (now - self.config.streaming.buffer_capacity).datetime
        self.waveform_axes.set_xlim(plot_starttime, now.datetime)
        plot_starttime = mdates.date2num(plot_starttime)

        # Remove pick-times that are off-screen
        for seed_id in self.p_times.keys():
            self.p_times[seed_id] = [
                pick_time for pick_time in self.p_times[seed_id]
                if pick_time > plot_starttime]
        for seed_id in self.s_times.keys():
            self.s_times[seed_id] = [
                pick_time for pick_time in self.s_times[seed_id]
                if pick_time > plot_starttime]

        return [self.waveform_axes]

    def update_table(self):
        listener_events = self.listener.old_events
        if len(self.listener.old_events) == 0:
            return []
        listener_events = sorted(
            listener_events, key=lambda ev: ev.time, 
            reverse=True)[0:MOST_RECENT]
        event_ids = {ev.event_id for ev in listener_events}
        # if event_ids.issubset(self._events_in_table) and not self.config.plotting.refresh:
        if event_ids.issubset(self._events_in_table):
            # No new events
            return [self.table_ax]
        if CHIME and not event_ids.issubset(self._events_in_table):
            chime.info()
        cells = self.event_table.get_celld()
        for row_num in range(MOST_RECENT):
            try:
                event = listener_events[row_num]
            except IndexError:
                event = NONEVENT
            for col_num, attrib in enumerate(COLUMNS.keys()):
                cell = cells[(row_num + 1, col_num)]
                text = event.__getattribute__(attrib)
                if attrib == "time":
                    text = dt.datetime(
                        year=text.year, month=text.month, day=text.day,
                        hour=text.hour, minute=text.minute, second=text.second,
                        microsecond=text.microsecond, tzinfo=dt.timezone.utc)
                    text = text.astimezone(LOCALTZ)
                    text = text.strftime("%Y/%m/%d %H:%M:%S")
                elif attrib == "depth":
                    text = f"{text / 1000:.1f}"
                else:
                    text = f"{text:.1f}"
                # print(f"({row_num + 1}, {col_num}): {text}")
                cell.set_text_props(text=text, ha="center")
        self._events_in_table.update(event_ids)
        return [self.table_ax]

    def update(self, *args, **kwargs):
        artists = []
        tic = time.perf_counter()
        artists.extend(self.update_waveforms())
        toc = time.perf_counter()
        Logger.info(f"Waveform update took {toc - tic:.3f}s")
        if self.map_ax:
            # pass
            tic = time.perf_counter()
            artists.extend(self.update_map())
            toc = time.perf_counter()
            Logger.info(f"Updating map took {toc - tic:.3f}s")

            tic = time.perf_counter()
            artists.extend(self.update_table())
            toc = time.perf_counter()
            Logger.info(f"Updating table took {toc - tic:.3f}s")
        return artists

    def animate(self):
        animator = FuncAnimation(
            fig=self.fig, func=self.update,
            interval=self.config.plotting.update_interval, blit=True)
        return animator

    def show(self, full_screen: bool = True):
        self.initialise_plot()
        if full_screen:
            self.fig.canvas.manager.full_screen_toggle()
        ani = self.animate()
        plt.show(block=True)
        # Close the streamer when the plot closes
        self.streamer.background_stop()
        self.listener.background_stop()


if __name__ == "__main__":
    import doctest

    doctest.testmod()
