"""
Animated near-real-time plotting of streaming data.
"""

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import logging
import matplotlib
import time
import os
import datetime as dt

from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
import matplotlib.dates as mdates
from typing import Iterable
from obspy import UTCDateTime

from earthquake_viewer.config.config import Config


DEPTH_CMAP = "plasma_r"  # To do - make a normalized cmap and scale bar
Logger = logging.getLogger(__name__)
LOCALTZ = dt.datetime.now().astimezone().tzinfo


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
                nrows=configuration.earthquake_viewer.n_chans,
                ncols=ncols)
            self.map_ax = fig.add_subplot(
                gs[:, 0:map_width],
                projection=configuration.plotting.map_projection)
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
            gs = fig.add_gridspec(
                nrows=configuration.earthquake_viewer.n_chans, ncols=1)
        self.waveform_axes = {}
        row, lead_ax = 0, None
        for seed_id in configuration.earthquake_viewer.seed_ids:
            ax = fig.add_subplot(gs[row, map_width:], sharex=lead_ax)
            ax.yaxis.tick_right()
            ax.yaxis.set_label_position("right")
            ax.set_ylabel(seed_id, rotation="horizontal",
                          horizontalalignment="left")
            ax.set_yticks([])
            self.waveform_axes.update({seed_id: ax})
            if row == 0:
                lead_ax = ax
            if row != len(configuration.earthquake_viewer.seed_ids) - 1:
                [label.set_visible(False) for label in ax.get_xticklabels()]
            # else:
            #     ax.xaxis.set_animated(True)
            row += 1
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
            key: UTCDateTime(1970, 1, 1) for key in self.waveform_axes.keys()}
        # Define artists
        self.map_scatters = None
        self.waveform_lines = {key: None for key in self.waveform_axes.keys()}
        # Start background services
        if not self.listener.busy:
            Logger.info("Starting event listening service")
            self.listener.background_run(event_type="earthquake")
        if not self.streamer.busy:
            Logger.info("Starting waveform streaming service")
            self.streamer.background_run()
        return

    def initialise_plot(self):
        if self.map_ax:
            self._initialise_map()
        # Initialise empty waveforms
        for seed_id in self.config.earthquake_viewer.seed_ids:
            Logger.info(f"Initialising for {seed_id}")
            ax = self.waveform_axes[seed_id]
            line = Line2D([0], [0], linewidth=0.5)  #, color="k")
            ax.add_line(line)
            self.waveform_lines.update({seed_id: line})
        final_ax = self.waveform_axes[
            self.config.earthquake_viewer.seed_ids[-1]]
        # Format ticks
        # minutes = mdates.MinuteLocator(tz=LOCALTZ)
        # seconds = mdates.SecondLocator(15, tz=LOCALTZ)
        tickformat =mdates.DateFormatter("%H:%M", tz=LOCALTZ)
        # final_ax.xaxis.set_major_locator(minutes)
        final_ax.xaxis.set_major_formatter(tickformat)
        # final_ax.xaxis.set_minor_locator(seconds)
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
            # self.map_ax.set_facecolor("white")  # oceans
            if self.config.plotting.latitude_range < 3:
                resolution = "h"
            elif self.config.plotting.latitude_range < 15:
                resolution = "i"
            else:
                resolution = "l"
            coast = cfeature.GSHHSFeature(
                scale=resolution, levels=[1], facecolor="yellowgreen",
                edgecolor="black", alpha=0.5)
            self.map_ax.add_feature(coast)
        self.map_ax.gridlines(draw_labels=True)
        if self._station_locations is not None:
            self.map_ax.scatter(
                [val[0] for val in self._station_locations.values()],
                [val[1] for val in self._station_locations.values()],
                marker="^", color="red", zorder=100,
                transform=ccrs.PlateCarree())
            for station_name, location in self._station_locations.items():
                self.map_ax.text(
                    location[0], location[1], station_name,
                    horizontalalignment='right',
                    verticalalignment="bottom", transform=ccrs.PlateCarree())
        # Make an empty scatter artist

        # Magnitude scale

        # Colorbar


        return

    def update_map(self):
        """ Get new events and plot. """
        listener_events = self.listener.old_events
        new_events = [ev for ev in listener_events
                      if ev.event_id not in self._plotted_event_ids]
        # Plot the new events!
        lats = [ev.latitude for ev in new_events]
        lons = [ev.longitude for ev in new_events]
        depths = [ev.depth for ev in new_events]
        mags = [ev.magnitude for ev in new_events]
        # Update the content of the artist!

        # Update alpha of old events?

        # Remove very old events.

        return self.map_scatters

    def update_waveforms(self):
        # Get data from buffers
        now = UTCDateTime.now()
        if not self.streamer.has_new_data:
            final_ax = self.waveform_axes[
                self.config.earthquake_viewer.seed_ids[-1]]
            final_ax.set_xlim(
                (now - self.config.streaming.buffer_capacity).datetime,
                now.datetime)
            return self.waveform_axes.values()
        stream = self.streamer.stream.copy()

        Logger.debug(stream)
        for tr in stream:
            seed_id = tr.id
            plot_lim = self._previous_plot_time[seed_id]
            if tr.stats.endtime <= plot_lim:
                continue  # No new data
            tic = time.time()
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
            tr = tr.merge()[0]
            toc = time.time()
            Logger.info(f"Filtering took {toc-tic:.3f}s")
            self._previous_plot_time.update({seed_id: tr.stats.endtime})
            tic = time.time()
            times = tr.times("matplotlib")
            data = tr.data
            toc = time.time()
            Logger.info(f"Getting times took {toc - tic:.3f}s")
            # Update!
            self.waveform_lines[seed_id].set_data(times, data)
            self.waveform_axes[seed_id].set_ylim(data.min(), data.max())

        # Update limit
        final_ax = self.waveform_axes[
            self.config.earthquake_viewer.seed_ids[-1]]
        final_ax.set_xlim(
            (now - self.config.streaming.buffer_capacity).datetime,
            now.datetime)

        return self.waveform_axes.values()

    def update(self, *args, **kwargs):
        Logger.debug("Updating")
        tic = time.time()
        artists = []
        artists.extend(self.update_waveforms())
        if self.map_ax:
            pass
            # artists.extend(self.update_map())
        toc = time.time()
        Logger.info(f"Update took {toc - tic:.3f}")
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



if __name__ == "__main__":
    import doctest

    doctest.testmod()
