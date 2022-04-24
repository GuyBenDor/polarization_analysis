import obspy
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
import math
from itertools import chain
from functions.baseFunctions import get_map_layers, haversine

batch_all = "AMID,ALMT,BLGI,DSHN,ENGV,LVBS,RSPN,KSH0,IZRL,GNNR,MLDT,OFRI,HNTI,ZRT,BTOR,KDZV,HDNS,RMOT,KNHM," \
    "GOSH,NATI,MSAM,HULT,TVR,MGDL,SPR,SHGL,GEM,GLHS,MLDT"
batch_all = (",").join(list(set(batch_all.split(","))))

site_file = "../data/sites.csv"
origin_file = "../data/relocated3.csv"
save_path = "dataframes/data1.csv"

minmax_time = [120,120]
max_dist = 10000
plot_locations = True

sites = pd.read_csv(site_file)
origins = pd.read_csv(origin_file, dtype={"evid": str})
origins["UTCtime"] = pd.to_datetime(origins["UTCtime"], format="%Y-%m-%dT%H:%M:%S.%fZ")
origins["starttime"] = origins["UTCtime"] - pd.Timedelta(seconds=minmax_time[0])
origins["endtime"] = origins["UTCtime"] + pd.Timedelta(seconds=minmax_time[1])


def plot_ev_locations(stations,data):

    coordinates = [stations.longitude.min() - 2, stations.longitude.max() + 2,
                   stations.latitude.min() - 2, stations.latitude.max() + 2]
    fig = plt.figure(figsize=(10, 9))

    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(coordinates)
    ax.scatter(stations["longitude"], stations["latitude"], transform=ccrs.PlateCarree(),
               s=160, zorder=3, label='station', color="yellow", marker='^', linewidths=0.5, edgecolors='k')

    ax.scatter(data["lon"], data['lat'], transform=ccrs.PlateCarree(),
               s=80, zorder=2, color="k",
               marker=".", linewidths=0.5, edgecolors='k')

    fname = "/Users/guy.bendor/Natural Earth Map/map_files/ME_Shaded6.nc"
    fault_file = "/Users/guy.bendor/Natural Earth Map/map_files/SomeLayers4Guy/MainFaultsMS.shp"
    water_file = "/Users/guy.bendor/Natural Earth Map/map_files/SomeLayers4Guy/WaterBodiesSubset.shp"
    border_file = "/Users/guy.bendor/Natural Earth Map/map_files/SomeLayers4Guy/WorldBorders.shp"
    faults, water, da, border = get_map_layers(fault_file,water_file,fname,border_file)
    da = da.sel(x=slice(coordinates[0], coordinates[1]), y=slice(coordinates[3], coordinates[2]))

    ax.imshow(da.__xarray_dataarray_variable__.data[0], transform=ccrs.PlateCarree(), origin='upper',
              cmap="Greys_r", extent=coordinates, zorder=0, alpha=0.6)

    ax.add_feature(faults, facecolor='None', edgecolor='r', linestyle='--', zorder=1)

    ax.add_geometries(water, facecolor='lightcyan', edgecolor='black', linestyle='-',
                      linewidth=0.5, crs=ccrs.PlateCarree(), zorder=1)

    ax.add_feature(border, facecolor='None', zorder=1, edgecolor="k", linestyle=':',
                   linewidth=0.5)
    ax.legend()
    gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=0.5, color='black', alpha=0.3, linestyle='--', draw_labels=True)
    gl.top_labels = True
    gl.left_labels = True
    gl.right_labels = True
    gl.xlocator = mticker.FixedLocator(np.arange(np.floor(coordinates[0]), coordinates[1] + 2, 2))
    gl.ylocator = mticker.FixedLocator(np.arange(np.floor(coordinates[2]), coordinates[3] + 2, 2))
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER

# save_path = f"dataframes/data_{val}.csv"
dfs = []

for station in batch_all.split(","):
    sta_inf = sites[sites.station == station].iloc[0]
    temp = origins.copy()

    lon_lat = [sta_inf.longitude, sta_inf.latitude]
    temp["epi_dist"] = temp[['lon', 'lat']].apply(lambda x: haversine(x[0], x[1], lon_lat[0], lon_lat[1]), axis=1)
    temp = temp[temp.epi_dist <= max_dist]
    temp["station"] = station
    temp.reset_index(drop=True, inplace=True)
    dfs.append(temp)
data = pd.concat(dfs)[["evid", "starttime", "endtime", "station", "lon", "lat"]]
data["station"] = data.groupby("evid")['station'].transform(lambda x: ','.join(sorted(set(x))))
data = data.drop_duplicates()
data.sort_values(by=["evid"], inplace=True)
data.reset_index(inplace=True, drop=True)
#!!!!!!!!!!!!!!!!!!
checker = True
itr = 0
dcopy = data.copy()
print(dcopy)
# dcopy["stations"] = globals()[val]
# for t in ["starttime", "endtime"]:
dcopy["day_of_year"] = dcopy.starttime.dt.dayofyear
dcopy[["starttime", "endtime"]] = dcopy.apply(
    lambda x: [obspy.UTCDateTime(x["starttime"]), obspy.UTCDateTime(x["endtime"])],
    axis = 1,
    result_type = "expand"
)
ev_list = dcopy.evid.str.split(",").tolist()
ev_list = list(chain.from_iterable(ev_list))#[ev_list.]
print(f"num of events in df: {len(ev_list)}")
print(f"num of unique events in df: {len(set(ev_list))}")

dcopy.to_csv(save_path, index=False)
# print(dcopy)
if plot_locations:
    stations = sites[sites.station.isin(batch_all.split(","))]
    plot_ev_locations(stations, data)
plt.show()
