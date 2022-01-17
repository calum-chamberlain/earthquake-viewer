"""
Configuration of the Earthquake-viewer
"""

import logging
import importlib
import os
import sys
import cartopy.crs as ccrs

from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:  # pragma: no cover
    from yaml import Loader, Dumper
from logging.handlers import RotatingFileHandler

from obspy import UTCDateTime
from obspy.core.util import AttribDict
from obspy.core.inventory import Inventory

from earthquake_viewer.listener import CatalogListener
from earthquake_viewer.streaming.streaming import _StreamingClient

Logger = logging.getLogger(__name__)


def seed_id_to_kwargs(seed_id: str) -> dict:
    keys = ("network", "station", "location", "channel")
    return {key: value for key, value in zip(keys, seed_id.split('.'))}


class _ConfigAttribDict(AttribDict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def to_yaml_dict(self):
        return {
            key.replace("_", " "): value
            for key, value in self.__dict__.items()}

    def __eq__(self, other):
        if set(self.__dict__.keys()) != set(other.__dict__.keys()):
            return False
        for key in self.__dict__.keys():
            if self[key] != other[key]:
                return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)


class EarthquakeViewerConfig(_ConfigAttribDict):
    defaults = {
        "seed_ids": ["NZ.WIZ.10.HHZ", "NZ.PUZ.10.HHZ", "NZ.OTVZ.10.HHZ",
                     "NZ.PXZ.10.HHZ", "NZ.WEL.10.HHZ",
                     "NZ.KHZ.10.HHZ",
                     "NZ.INZ.10.HHZ",
                     "NZ.MQZ.10.HHZ",
                     "NZ.JCZ.10.HHZ", "NZ.OPZ.10.HHZ", "NZ.PYZ.10.HHZ"],
    }

    readonly = []

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def n_chans(self):
        return len(self.seed_ids)


class PlottingConfig(_ConfigAttribDict):
    defaults = {
        "figure_size": (10, 10),
        "lowcut": 1.0,
        "highcut": 10.0,
        "decimate": 1,
        "backend": "matplotlib",
        "style": "eqv.mplstyle",
        "update_interval": 40,
        "plot_map": True,
        "map_width_percent": 30,
        "map_client": "GEONET",
        "map_client_type": "FDSN",
        "map_update_interval": 120,
        "map_bounds": (165.96, 180, -47.23, -34.54),
        "event_history": 86400 * 2,
        "label_stations": True,
        "refresh": True,
    }
    readonly = []

    client_base = "obspy.clients"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def client_module(self):
        return importlib.import_module(
            f"{self.client_base}.{self.map_client_type.lower()}")

    def get_client(self):
        """ Get the client instance given the set parameters. """
        try:
            _client_module = self.client_module
        except ModuleNotFoundError as e:
            Logger.error(e)
            return None
        try:
            client = _client_module.Client(self.map_client)
        except Exception as e:
            Logger.error(e)
            return None
        return client

    @property
    def longitude_range(self):
        if self.map_bounds is None:
            return None
        try:
            return self.map_bounds[1] - self.map_bounds[0]
        except Exception as e:
            print(e)
        return None

    @property
    def latitude_range(self):
        if self.map_bounds is None:
            return None
        try:
            return self.map_bounds[3] - self.map_bounds[2]
        except Exception as e:
            print(e)
        return None

    @property
    def global_map(self):
        if self.longitude_range > 30 and self.latitude_range > 30:
            return True
        return False

    @property
    def map_projection(self):
        if self.global_map:
            return ccrs.PlateCarree()
        return ccrs.AlbersEqualArea(
            central_longitude=self.map_bounds[0] + 0.5 * self.longitude_range,
            central_latitude=self.map_bounds[2] + 0.5 * self.latitude_range,
            standard_parallels=[self.map_bounds[2], self.map_bounds[3]])

    def get_listener(self, populate: bool = True) -> CatalogListener:
        """
        Get the listener service

        Parameters
        ----------
        populate
            Whether to populate the listener with old events or not
        """
        if not self.plot_map:
            return None
        client = self.get_client()
        catalog_lookup_kwargs = dict(
            minlatitude=self.map_bounds[2],
            minlongitude=self.map_bounds[0],
            maxlatitude=self.map_bounds[3],
            maxlongitude=self.map_bounds[1])
        if populate:
            now = UTCDateTime.now()
            try:
                catalog = client.get_events(
                    starttime=now - self.event_history, endtime=now,
                    **catalog_lookup_kwargs)
            except Exception as e:
                print(f"Could not populate catalog due to {e}")
                catalog = None
        else:
            catalog = None
        return CatalogListener(
            client=client, catalog_lookup_kwargs=catalog_lookup_kwargs,
            interval=self.map_update_interval, keep=self.event_history,
            catalog=catalog, refresh=self.refresh)


class StreamingConfig(_ConfigAttribDict):
    defaults = {
        "rt_client_url": "link.geonet.org.nz",
        "rt_client_type": "seedlink",
        "buffer_capacity": 600.,
    }
    readonly = []
    rt_client_base = "earthquake_viewer.streaming.clients"
    _known_keys = {"starttime", "query_interval", "speed_up", "client_type"}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @property
    def rt_client_module(self):
        return importlib.import_module(
            f"{self.rt_client_base}.{self.rt_client_type.lower()}")

    @property
    def known_kwargs(self):
        out = {}
        for key in self._known_keys:
            value = self.get(key, None)
            if value is not None:
                out.update({key: value})
        return out

    def get_streaming_client(self) -> _StreamingClient:
        """ Get the configured waveform streaming service. """
        try:
            _client_module = self.rt_client_module
        except ModuleNotFoundError as e:
            Logger.error(e)
            return None
        try:
            kwargs = self.known_kwargs
            rt_client = _client_module.RealTimeClient(
                server_url=self.rt_client_url,
                buffer_capacity=self.buffer_capacity,
                **kwargs)
        except Exception as e:
            Logger.error(e)
            return None
        return rt_client


KEY_MAPPER = {
    "earthquake_viewer": EarthquakeViewerConfig,
    "plotting": PlottingConfig,
    "streaming": StreamingConfig,
}


class Config(object):
    """
    Base configuration parameters from Earthquake Viewer.

    Parameters
    ----------
    log_level
        Any parsable string for logging.basicConfig
    log_formatter
        Any parsable string formatter for logging.basicConfig
    earthquake_viewer
        Config values for real-time earthquake-viewer
    plotting
        Config values for real-time plotting
    streaming
        Config values for real-time streaming
    """
    def __init__(
        self,
        log_level: str = "INFO",
        log_formatter: str = "%(asctime)s\t[%(processName)s:%(threadName)s]: %(name)s\t%(levelname)s\t%(message)s",
        **kwargs
    ):
        self.earthquake_viewer = EarthquakeViewerConfig()
        self.plotting = PlottingConfig()
        self.streaming = StreamingConfig()
        self.log_level = log_level
        self.log_formatter = log_formatter

        for key, value in kwargs.items():
            if key not in KEY_MAPPER.keys():
                raise NotImplementedError("Unsupported argument "
                                          "type: {0}".format(key))
            if isinstance(value, dict):
                self.__dict__[key] = KEY_MAPPER[key](value)
            else:
                assert isinstance(value, type(self.__dict__[key]))
                self.__dict__[key] = value

    def __repr__(self):
        return ("Config(\n\tearthquake_viewer={0},\n\tplot={1},\n\t"
                "streaming={2}".format(
                    self.earthquake_viewer.__repr__(), self.plotting.__repr__(),
                    self.streaming.__repr__()))

    def __eq__(self, other):
        if not isinstance(other, Config):
            return False
        if set(self.__dict__.keys()) != set(other.__dict__.keys()):
            return False
        for key in self.__dict__.keys():
            if not self.__dict__[key] == other.__dict__[key]:
                return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    def write(self, config_file: str) -> None:
        """
        Write the configuration to a tml formatted file.

        Parameters
        ----------
        config_file
            path to the configuration file. Will overwrite and not warn
        """
        with open(config_file, "w") as f:
            f.write(dump(self.to_yaml_dict(), Dumper=Dumper))

    def to_yaml_dict(self) -> dict:
        """ Make a more human readable yaml format """
        _dict = {}
        for key, value in self.__dict__.items():
            if hasattr(value, "to_yaml_dict"):
                _dict.update({key: value.to_yaml_dict()})
            else:
                _dict.update({key: value})
        return _dict

    def get_inventory(self, level: str = "station") -> Inventory:
        """ The inventory given the seed ids. """
        if len(self.earthquake_viewer.seed_ids) == 0:
            return Inventory()
        client = self.plotting.get_client()
        inv = client.get_stations(
            level=level,
            **seed_id_to_kwargs(self.earthquake_viewer.seed_ids[0]))
        if len(self.earthquake_viewer.seed_ids) > 1:
            for seed_id in self.earthquake_viewer.seed_ids[1:]:
                inv += client.get_stations(
                    level=level, **seed_id_to_kwargs(seed_id))
        return inv

    def get_listener(self) -> CatalogListener:
        return self.plotting.get_listener()

    def get_streamer(self) -> _StreamingClient:
        streamer = self.streaming.get_streaming_client()
        # Add channels to streamer
        for seed_id in self.earthquake_viewer.seed_ids:
            selector = seed_id_to_kwargs(seed_id)
            selector.pop("location")
            selector.update({"selector": selector.pop("channel")})
            selector.update({"net": selector.pop("network")})
            try:
                streamer.select_stream(**selector)
            except Exception as e:
                Logger.error(f"Could not add {seed_id} due to {e}")
        return streamer

    def setup_logging(
        self,
        screen: bool = True,
        file: bool = True,
        filename: str = "earthquake-viewer.log",
        **kwargs
    ):
        """Set up logging using the logging parameters."""
        handlers = []
        if file:
            file_log_args = dict(filename=filename, mode='a',
                                 maxBytes=20*1024*1024, backupCount=2,
                                 encoding=None, delay=0)
            file_log_args.update(kwargs)
            rotating_handler = RotatingFileHandler(**file_log_args)
            rotating_handler.setFormatter(
                logging.Formatter(self.log_formatter))
            rotating_handler.setLevel(self.log_level)
            handlers.append(rotating_handler)
        if screen:
            # Console handler
            console_handler = logging.StreamHandler(stream=sys.stdout)
            console_handler.setLevel(self.log_level)
            console_handler.setFormatter(
                logging.Formatter(self.log_formatter))
            handlers.append(console_handler)
        logging.basicConfig(
            level=self.log_level, format=self.log_formatter,
            handlers=handlers)


def read_config(config_file=None) -> Config:
    """
    Read configuration from a yml file.

    Parameters
    ----------
    config_file
        path to the configuration file.

    Returns
    -------
    Configuration with required defaults filled and updated based on the
    contents of the file.
    """
    if config_file is None:
        return Config()
    if not os.path.isfile(config_file):
        raise FileNotFoundError(config_file)
    with open(config_file, "rb") as f:
        configuration = load(f, Loader=Loader)
    config_dict = {}
    for key, value in configuration.items():
        if key.replace(" ", "_") in KEY_MAPPER.keys():
            config_dict.update(
                {key.replace(" ", "_"):
                     {_key.replace(" ", "_"): _value
                      for _key, _value in value.items()}})
        else:
            config_dict.update({key: value})
    return Config(**config_dict)


if __name__ == "__main__":
    import doctest

    doctest.testmod()
