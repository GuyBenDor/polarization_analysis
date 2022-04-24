import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy import geodesic
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import cartopy.io.shapereader as shpreader
import xarray as xr
from shapely.ops import unary_union
import matplotlib.ticker as mticker
from functions.baseFunctions import get_map_layers

site_path = "../data/sites.csv"
event_path = "../data/relocated3.csv"

ev = ['123379', '123622', '123624', '123703']#,"125796"]
sites = pd.read_csv(site_path)
events = pd.read_csv(event_path,dtype={"evid":str})

events = events[events.evid.isin(ev)]

# Kinneret - coordinates
min_lat, max_lat = 32.7, 32.9
min_lon, max_lon = 35.5, 35.65

rectangle = [min_lon - 0.25, max_lon + 0.25,min_lat - 0.25, max_lat + 0.25]

sites = sites[
    (sites.longitude>=rectangle[0]) &
    (sites.longitude<=rectangle[1]) &
    (sites.latitude>=rectangle[2]) &
    (sites.latitude<=rectangle[3])
]
fname = "/Users/guy.bendor/Natural Earth Map/map_files/ME_Shaded6.nc"
fault_file = "/Users/guy.bendor/Natural Earth Map/map_files/SomeLayers4Guy/MainFaultsMS.shp"
water_file = "/Users/guy.bendor/Natural Earth Map/map_files/SomeLayers4Guy/WaterBodiesSubset.shp"
border_file = "/Users/guy.bendor/Natural Earth Map/map_files/SomeLayers4Guy/WorldBorders.shp"
faults, water, da, border = get_map_layers(fault_file, water_file, fname, border_file)

da = da.sel(x=slice(rectangle[0], rectangle[1]), y=slice(rectangle[3], rectangle[2]))

fig = plt.figure(figsize=(7, 5))
# ax = fig.add_subplot(1, 1, 1, projection=request.crs)
# ax = fig.add_subplot(1, 1, 1, projection=request.crs)
ax = plt.axes(projection=ccrs.PlateCarree())

ax.set_extent(rectangle, crs=ccrs.PlateCarree())

ax.scatter(sites["longitude"], sites["latitude"],
           transform=ccrs.PlateCarree(), s=100, zorder=4,
           label='scatter', color='yellow', marker='^',
           linewidths=0.5, edgecolors='k')

ax.scatter(events["lon"], events["lat"],
           transform=ccrs.PlateCarree(), s=100, zorder=4,
           label='scatter', color='r', marker='.',
           linewidths=0.5, edgecolors='k')
for i in range(len(sites["station"])):
    ax.text(sites.iloc[i]["longitude"], sites.iloc[i]["latitude"] + 0.01, sites.iloc[i]["station"], transform=ccrs.PlateCarree())

ax.imshow(da.__xarray_dataarray_variable__.data[0], transform=ccrs.PlateCarree(), origin='upper',
          cmap="Greys_r", extent=rectangle, zorder=0, alpha=0.6)

ax.add_feature(faults, facecolor='None', edgecolor='r', linestyle='--', zorder=2)

ax.add_geometries(water, facecolor='lightcyan', edgecolor='black', linestyle='-',
                  linewidth=0.5, crs=ccrs.PlateCarree(), zorder=1)

ax.add_feature(border, facecolor='None', zorder=1, edgecolor="k",linestyle=':',
                  linewidth=0.5)

gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=0.5, color='black', alpha=0.3, linestyle='--', draw_labels=True)
# gl.xlabels_top = True
# gl.ylabels_left = True
# gl.ylabels_right = True
gl.xlocator = mticker.FixedLocator(np.arange(0, 50, 1))
gl.ylocator = mticker.FixedLocator(np.arange(0, 50, 1))
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

plt.show()