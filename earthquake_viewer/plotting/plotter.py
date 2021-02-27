"""
Animated near-real-time plotting of streaming data.
"""

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import logging

from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D
from typing import Iterable
from obspy import UTCDateTime

from earthquake_viewer.config.config import Config


DEPTH_CMAP = "plasma_r"  # To do - make a normalized cmap and scale bar
Logger = logging.getLogger(__name__)


def _scale_mags(magnitudes: Iterable) -> Iterable:
    return [m ** 3 for m in magnitudes]


class Plotter(object):
    def __init__(
        self,
        configuration: Config,
    ):
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
            self.map_ax, self._station_locations = None, None
            map_width = 0
            gs = fig.add_gridspec(
                nrows=configuration.earthquake_viewer.n_chans, ncols=1)
        self.waveform_axes = {}
        row, lead_ax = 0, None
        for seed_id in configuration.earthquake_viewer.seed_ids:
            self.waveform_axes.update({
                seed_id: fig.add_subplot(gs[row, map_width:], sharex=lead_ax)})
            if row == 0:
                lead_ax = self.waveform_axes[seed_id]
            row += 1
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
            ax = self.waveform_axes[seed_id]
            line = Line2D([0], [0], linewidth=1.0, color="k")
            ax.add_line(line)
            self.waveform_lines.update({seed_id: line})
        return self.fig

    def _initialise_map(self):
        # Make the blank map!
        if self.config.plotting.global_map:
            self.map_ax.set_global()
            self.map_ax.stock_img()  # Plot a nice image on the globe
        else:
            self.map_ax.set_extent(self.config.plotting.map_bounds,
                                   crs=ccrs.PlateCarree())
            self.map_ax.set_facecolor("white")  # oceans
            if self.config.plotting.latitude_range < 3:
                resolution = "h"
            elif self.config.plotting.latitude_range < 10:
                resolution = "i"
            else:
                resolution = "l'"
            coast = cfeature.GSHHSFeature(
                scale=resolution, levels=[1], facecolor="lightgrey",
                edgecolor="black")
            self.map_ax.add_feature(coast)
        self.map_ax.gridlines(draw_labels=True)
        if self._station_locations is not None:
            self.map_ax.scatter(
                [val[0] for val in self._station_locations.values()],
                [val[1] for val in self._station_locations.values()],
                marker="^", color="red",
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
        stream = self.streamer.stream.copy()
        for tr in stream:
            seed_id = tr.id
            plot_lim = self._previous_plot_time[seed_id]
            if tr.stats.endtime <= plot_lim:
                continue  # No new data
            self.waveform_lines.set_data(tr.times("matplotlib"), tr.data)

        # Update limit

        return self.waveform_lines

    def update(self, *args, **kwargs):
        artists = []
        artists.extend(self.update_waveforms())
        if self.map_ax:
            artists.extend(self.update_map())
        # If using blitting (which should be faster!) then this needs to return an iterable of updated artists
        return artists

    def animate(self):
        animator = FuncAnimation(
            fig=self.fig, func=self.update, init_func=self.initialise_plot,
            interval=1000, blit=True)
        return animator

    def show(self, full_screen: bool = True):
        if full_screen:
            self.fig.canvas.manager.full_screen_toggle()
        self.fig.show()


if __name__ == "__main__":
    import doctest

    doctest.testmod()
