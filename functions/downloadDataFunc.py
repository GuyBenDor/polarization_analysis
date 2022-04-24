import os
os.environ['CONDA_PREFIX'] = ''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy import geodesic
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
import math
import time

from functions.baseFunctions import haversine
from obspy.clients.fdsn import Client
from functions.baseFunctions import sta_info, stations, events
import cartopy.io.shapereader as shpreader
import xarray as xr
from shapely.ops import cascaded_union
from shapely.geometry import Polygon


url = "http://82.102.143.46:8181"
client = Client(base_url=url, user='test', password='test')

colors = {
    np.nan: {"val": " ", "s": 5, "c": "red"},
    0: {"val": "  < 2", "s": 10, "c": "red"},
    1: {"val": "  < 2", "s": 10, "c": "red"},
    2: {"val": "2 < 3", "s": 20, "c": "green"},
    3: {"val": "3 < 4", "s": 30, "c": "purple"},
    4: {"val": "4 <  ", "s": 40, "c": "blue"},
}



# def map_plot(df,**kwargs):
#     dx = 1
#     min_lon, min_lat = (df[["lon", "lat"]].min() - dx).values
#     max_lon, max_lat = (df[["lon", "lat"]].max() + dx).values
#     coordinates = [min_lon, max_lon, min_lat, max_lat]
#     ev = df.loc[df["evid"].notna()]
#     st = df.loc[df["station"].notna()]
#     title = ev["evid"].iloc[0] if len(ev["evid"]) == 1 else st["station"].iloc[0]
#
#     fig = plt.figure(figsize=(10, 9))
#
#     ax = plt.axes(projection=ccrs.PlateCarree())
#     ax.set_extent(coordinates)
#
#     ax.scatter(st["lon"], st["lat"], transform=ccrs.PlateCarree(), s=80, zorder=3, label='station', color="yellow",
#                marker='^', linewidths=0.5, edgecolors='k')
#     for label in ev.val.unique():
#         temp = ev.loc[ev.val == label]
#         ax.scatter(temp["lon"], temp['lat'], transform=ccrs.PlateCarree(), s=temp["s"], zorder=3, label=label,
#                    color=temp["c"], marker="o", linewidths=0.5, edgecolors='k'
#                    )
#
#     if "ev_out" in kwargs:
#         ax.scatter(kwargs["ev_out"]["lon"], kwargs["ev_out"]['lat'], transform=ccrs.PlateCarree(),
#                    s=kwargs["ev_out"]["s"], zorder=3, label="out of scope", color=kwargs["ev_out"]["c"],
#                    marker=".", linewidths=0.5, edgecolors='k'
#                    )
#
#     if len(ev["evid"]) == 1:
#         for i in range(len(st["station"])):
#             ax.text(st["lon"].iloc[i], st["lat"].iloc[i], st["station"].iloc[i], transform=ccrs.PlateCarree(), zorder=3)
#
#         for radi in ["minDist","maxDist"]:
#             if radi in kwargs:
#                 circle_points = geodesic.Geodesic().circle(lon=ev["lon"][0], lat=ev["lat"][0],
#                                                            radius=kwargs[radi]*1000, n_samples=200, endpoint=False)
#                 geom = Polygon(circle_points)
#                 ax.add_geometries((geom,), crs=ccrs.PlateCarree(), facecolor='none',
#                                   edgecolor='k', linewidth=1, zorder=2)
#
#     fname = "/Users/guy.bendor/Natural Earth Map/map_files/ME_Shaded6.tif"
#     da = xr.open_rasterio(fname).sel(x=slice(coordinates[0], coordinates[1]), y=slice(coordinates[3], coordinates[2]))
#     ax.imshow(da.variable.data[0], transform=ccrs.PlateCarree(), origin='upper',
#               cmap="Greys_r", extent=coordinates, zorder=0, alpha=0.6)
#
#     fault_file = "/Users/guy.bendor/Natural Earth Map/map_files/SomeLayers4Guy/MainFaultsMS.shp"
#     faults = cfeature.ShapelyFeature(shpreader.Reader(fault_file).geometries(), ccrs.PlateCarree())
#     ax.add_feature(faults, facecolor='None', edgecolor='r', linestyle='--', zorder=2)
#
#     water_file = "/Users/guy.bendor/Natural Earth Map/map_files/SomeLayers4Guy/WaterBodiesSubset.shp"
#     water = cfeature.ShapelyFeature(shpreader.Reader(water_file).geometries(), ccrs.PlateCarree())
#     water = cascaded_union(list(water.geometries()))
#     ax.add_geometries(water, facecolor='lightcyan', edgecolor='black', linestyle='-',
#                       linewidth=0.5, crs=ccrs.PlateCarree(), zorder=1)
#
#     border_file = "/Users/guy.bendor/Natural Earth Map/map_files/SomeLayers4Guy/WorldBorders.shp"
#     border = cfeature.ShapelyFeature(shpreader.Reader(border_file).geometries(), ccrs.PlateCarree())
#     ax.add_feature(border, facecolor='None', zorder=1, edgecolor="k", linestyle=':',
#                    linewidth=0.5)
#     ax.legend()
#     gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=0.5, color='black', alpha=0.3, linestyle='--', draw_labels=True)
#     gl.xlabels_top = True
#     gl.ylabels_left = True
#     gl.ylabels_right = True
#     gl.xlocator = mticker.FixedLocator(np.arange(np.floor(min_lon), max_lon + 2, 2))
#     gl.ylocator = mticker.FixedLocator(np.arange(np.floor(min_lat), max_lat + 2, 2))
#     gl.xformatter = LONGITUDE_FORMATTER
#     gl.yformatter = LATITUDE_FORMATTER
#     plt.suptitle(title)


class site(sta_info, events):
    """
    Class method to generate a site_dist objects which inherits its properties from the site class.
    1. The class extracts information such as matching lon, lat and elevation given a station name.
    2. A method exists for adding a dataframe containing all epicentral distances.
    ----A site_file must be pre-defined as a class method----
    ----A origin_file must be pre-defined as a class method----
    """
    num_of_sites = 0

    def __init__(self, *args):
        if len(args) == 1:
            super().__init__(args[0])
            self.set_distance_df()
            site.num_of_sites += 1
        else:
            print("no event was defined")
            pass

    @classmethod
    def set_min_max_dist(cls, **kwargs):
        if 'minDist' in kwargs: cls.minDist = kwargs['minDist']
        if 'maxDist' in kwargs: cls.maxDist = kwargs['maxDist']

    def set_distance_df(self):
        self.ev_df["distance"] = self.ev_df[["lon", "lat"]].apply(
            lambda x: haversine(self.lon, self.lat, x[0], x[1]), axis=1)
        df = self.ev_df.copy()
        if hasattr(self, 'minDist'): df = df[df.distance >= self.minDist]
        if hasattr(self, 'maxDist'): df = df[df.distance < self.maxDist]
        df.sort_values(by=["distance"], inplace=True, ignore_index=True)
        self.distance_df = df

    def plot_map_view(self, in_range=True):
        st = {"station": [self.name], "lon": [self.lon], "lat": [self.lat]}
        st = pd.DataFrame(st)
        ev = self.ev_df.copy()

        if in_range:
            if hasattr(self, 'minDist'):
                ev.loc[ev.distance < self.minDist, "mag"] = np.nan
            if hasattr(self, 'maxDist'):
                ev.loc[ev.distance > self.maxDist, "mag"] = np.nan
        ev["mag"] = ev["mag"].apply(np.floor)
        ev["color"] = ev["mag"].map(colors)
        df2 = pd.json_normalize(ev['color'])
        ev = pd.concat([ev, df2], axis=1)
        ev = ev[["evid", "lon", "lat", "mag", "val", "s", "c"]]
        ev2 = ev.loc[ev.mag.isna()]
        ev.dropna(inplace=True)
        df = pd.concat([ev, st], axis=0)
        # map_plot(df, ev_out=ev2)


class event(stations, events):
    num_of_events = 0

    def __init__(self, *args):
        if len(args) == 1:
            self._evid = args[0]
            self.set_event_data()
            event.num_of_events += 1
        else:
            print("no event was defined")
            pass

    @property
    def evid(self):
        return self._evid

    @evid.setter
    def evid(self, val):
        event.num_of_events += 1
        self._evid = val
        self.set_event_data()

    @classmethod
    def set_min_max_dist(cls, **kwargs):
        if 'minDist' in kwargs: cls.minDist = kwargs['minDist']
        if 'maxDist' in kwargs: cls.maxDist = kwargs['maxDist']

    @classmethod
    def set_t_before_after(cls, *args):
        cls.t_before = args[0]*60.
        cls.t_after = args[1]*60.

    @classmethod
    def remove_stations(cls, string):
        cls.st_to_remove = string
        cls.sta_df = cls.sta_df[~cls.sta_df.station.isin(cls.st_to_remove.split(","))]
        cls.stations_kept = ",".join(cls.sta_df.station.tolist())
        # for val in args:
        #     stations.sta_df.drop(stations.sta_df.station==val, axis=0, inplace=True)

    @classmethod
    def keep_stations(cls, string):
        #cls.st_to_remove = l
        cls.stations_kept = string
        cls.sta_df = cls.sta_df[cls.sta_df.station.isin(cls.stations_kept.split(","))]

    def set_event_data(self):
        row = self.ev_df[self.ev_df.evid == self._evid].iloc[0]
        attrs = ['evname', 'lat', 'lon', 'depth', 'mb', 'ms', 'ml', 'originTime', 'mag']
        for att in attrs:
            setattr(self, att, row[att])

        self.starttime = self.originTime - self.t_before
        self.endtime = self.originTime + self.t_after
        self.get_station_in_range()

    def get_station_in_range(self):
        self.sta_df["distance"] = self.sta_df[["lon", "lat"]].apply(
            lambda x: haversine(x[0], x[1], self.lon, self.lat), axis=1)
        sta = self.sta_df.copy()
        if hasattr(self, "st_to_remove"): sta = sta[~sta.station.isin(self.st_to_remove)]
        if hasattr(self, 'minDist'): sta = sta[sta.distance >= self.minDist]
        if hasattr(self, 'maxDist'): sta = sta[sta.distance < self.maxDist]
        sta.sort_values(by=["distance", "station"], inplace=True, ignore_index=True)
        self.sta_in_range = ",".join(sta.station.values)
        self.sta_in_range_df = sta

    def plot_map_view(self, in_range=True, **kwargs):
        ev = {"evid": [self._evid], "lon": [self.lon], "lat": [self.lat]}
        ev.update(colors[np.floor(self.mag)])
        ev = pd.DataFrame(ev)

        st = self.sta_df
        st["st_in"] = True
        if in_range:
            if hasattr(self, 'minDist'):
                st.loc[st.distance < self.minDist, "st_in"] = False
                kwargs.update({"maxDist": self.minDist})
            if hasattr(self, 'maxDist'):
                st.loc[st.distance > self.maxDist, "st_in"] = False
                kwargs.update({"maxDist": self.maxDist})

        st_out = st[~st["st_in"]]
        st = st[st["st_in"]]
        df = pd.concat([ev, st], axis=0)

        # map_plot(df, **kwargs)















# class event(stations, events):
#     num_of_events = 0
#
#     def __init__(self, *args):
#         if len(args) == 1:
#             self._evid = args[0]
#             self.set_event_data()
#             event.num_of_events += 1
#         else:
#             print("no event was defined")
#             pass
#
#     @property
#     def evid(self):
#         return self._evid
#
#     @evid.setter
#     def evid(self, val):
#         event.num_of_events += 1
#         self._evid = val
#         self.set_event_data()
#
#     @classmethod
#     def set_min_max_dist(cls, **kwargs):
#         if 'minDist' in kwargs: cls.minDist = kwargs['minDist']
#         if 'maxDist' in kwargs: cls.maxDist = kwargs['maxDist']
#
#     @classmethod
#     def set_t_before_after(cls, *args):
#         cls.t_before = args[0]*60.
#         cls.t_after = args[1]*60.
#
#     @classmethod
#     def remove_stations(cls, l):
#         cls.st_to_remove = l
#         cls.sta_df = cls.sta_df[~cls.sta_df.station.isin(cls.st_to_remove)]
#         # for val in args:
#         #     stations.sta_df.drop(stations.sta_df.station==val, axis=0, inplace=True)
#
#     def set_event_data(self):
#         row = self.ev_df[self.ev_df.evid == self._evid].iloc[0]
#         attrs = ['evname', 'lat', 'lon', 'depth', 'mb', 'ms', 'ml', 'originTime', 'mag']
#         for att in attrs:
#             setattr(self, att, row[att])
#
#         self.starttime = self.originTime - self.t_before
#         self.endtime = self.originTime + self.t_after
#         self.get_station_in_range()
#
#     def get_station_in_range(self):
#         self.sta_df["distance"] = self.sta_df[["lon", "lat"]].apply(
#             lambda x: haversine(x[0], x[1], self.lon, self.lat), axis=1)
#         sta = self.sta_df.copy()
#         if hasattr(self, "st_to_remove"): sta = sta[~sta.station.isin(self.st_to_remove)]
#         if hasattr(self, 'minDist'): sta = sta[sta.distance >= self.minDist]
#         if hasattr(self, 'maxDist'): sta = sta[sta.distance < self.maxDist]
#         sta.sort_values(by=["distance", "station"], inplace=True, ignore_index=True)
#         self.sta_in_range = ",".join(sta.station.values)
#         self.sta_in_range_df = sta
#
#     def plot_map_view(self, in_range=True, **kwargs):
#         ev = {"evid": [self._evid], "lon": [self.lon], "lat": [self.lat]}
#         ev.update(colors[np.floor(self.mag)])
#         ev = pd.DataFrame(ev)
#
#         st = self.sta_df
#         st["st_in"] = True
#         if in_range:
#             if hasattr(self, 'minDist'):
#                 st.loc[st.distance < self.minDist, "st_in"] = False
#                 kwargs.update({"maxDist": self.minDist})
#             if hasattr(self, 'maxDist'):
#                 st.loc[st.distance > self.maxDist, "st_in"] = False
#                 kwargs.update({"maxDist": self.maxDist})
#
#         st_out = st[~st["st_in"]]
#         st = st[st["st_in"]]
#         df = pd.concat([ev, st], axis=0)
#
#         map_plot(df, **kwargs)

# class station_data:
#
#     def __init__(self, name):
#         self.name = name
#
#     @classmethod
#     def set_path_to_tables(cls, path):
#         cls.path = path
#
#     @classmethod
#     def set_time_before_after_orig(cls, start_minuntes, end_minutes):
#         cls.starttime = start_minuntes
#         cls.endtime = end_minutes
#
#     @property
#     def event_df(self):
#         station = pd.read_csv(os.path.join(self.path, self.name + ".csv"))
#         station.UTCtime = station.UTCtime.apply(obspy.UTCDateTime)
#         station["starttime"] = station.UTCtime - 60 * self.starttime
#         station["endtime"] = station.UTCtime + 60 * self.endtime
#         station = station[["evid", "starttime", "endtime"]]
#         return station
#
#     def __str__(self):
#         return f"Station.name: {self.name}\nStation.event_df: {len(self.event_df)}"


def get_event(station, stime, etime, channel="*", remove_response=False, pre_filter=None):
    st = client.get_waveforms("*", station, "*", channel, stime, etime, attach_response=True)
    st = st.merge()
    start_trim = max([s.stats.starttime for s in st])
    end_trim = min([s.stats.endtime for s in st])
    st.trim(start_trim, end_trim)
    if remove_response:
        if pre_filter:
            return st.remove_response(output="DISP", water_level=60, pre_filt=pre_filter)
        else:
            return st.remove_response(output="DISP", water_level=60)
    else:
        return st


def save_data(data, path_for_saving):
    start = time.perf_counter()

    file_path = os.path.join(path_for_saving, data.name)
    print(file_path)
    if not os.path.exists(file_path):
        os.makedirs(file_path)

    print(f"Number of events to download: {len(data.event_df)}")

    number_of_downloaded_events = 0

    existing_evids = [s.split(".")[4] for s in os.listdir(file_path)]
    print(existing_evids)
    for i in range(len(data.event_df)):

        row = data.event_df.iloc[i]
        print(f"number: {i} out of {len(data.event_df)}")
        if not str(row.evid) in existing_evids:
            try:
                st = get_event(data.name, row.starttime, row.endtime)
                for tr in st:
                    tr.write(os.path.join(file_path, f"{tr.id}.{str(row.evid)}.SAC"), format="SAC")
                    # st.plot()
                number_of_downloaded_events += 1
            except:
                continue

    end = time.perf_counter()
    print(f"Number of events you managed to download: {number_of_downloaded_events}")
    print(f"Finished in {round(end - start, 2)} second(s)")
