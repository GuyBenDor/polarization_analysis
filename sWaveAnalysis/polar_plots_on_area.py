import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy import geodesic
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
import cartopy.io.shapereader as shpreader
import xarray as xr
from shapely.ops import cascaded_union
from shapely.geometry import Polygon
import pandas as pd
import numpy as np
from functions.baseFunctions import haversine,initial_bearing,convert_ang_to_xy,get_map_layers
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator,NullFormatter)
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.projections import get_projection_class


event_path = "../data/relocated3.csv"
site_path = "../data/sites.csv"
dir_of_split_dat = "splitParam"

region = "kinneret" # hula, kinneret
hula_stations = ["AMID","DSHN","GEM","HDNS","KDZV","LVBS","MTLA","RSPN","KSH0"]
kinneret_stations = ["AMID","GLHS","HDNS","KNHM","MGDL","RMOT","TVR","RSPN","KSH0"]

if region=="hula":
    min_lat, max_lat = 32.9, 33.28
    min_lon, max_lon = 35.52, 35.7
    stations = hula_stations

    conditions = [
        ["GEM", [0.04, 0.14]],
        ["LVBS", [0.18, 0.085]],
        ["KDZV", [0.06, 0.03]],
        ["HDNS", [0.08, -0.12]],
        ["AMID", [-0.02, -0.13]],
        ["RSPN", [-0.17, -0.05]],
        ["DSHN", [-0.13, 0.04]],
        ["MTLA", [-0.15, 0.03]],
        ["KSH0", [0.05, -0.06]]
    ]
else:
    min_lat, max_lat = 32.7, 32.9
    min_lon, max_lon = 35.5, 35.65
    stations = kinneret_stations

    conditions = [
        ["RSPN", [-0.1, 0.0]],
        ["AMID", [-0.17, -0.06]],
        ["MGDL", [-0.1, -0.12]],
        ["TVR", [-0.01, -0.08]],
        ["GLHS", [0.02, -0.1]],
        ["RMOT", [0.08, -0.107]],
        ["HDNS", [0.11, -0.02]],
        ["KNHM", [0., 0.07]],
        ["KSH0", [-0.12, 0.07]]



    ]

min_rec, min_corr_coef = 0.85, 0.7
Type = '2d'



conditions = {i[0]: i[1] for i in conditions}

for st in stations:
    if st not in conditions.keys():
        conditions.update({st: [0, 0]})

arrow_dict = {'scale': 6.5, 'width': 0.002, 'headwidth': 1., 'headlength': 0.001, "scale_units": "inches", "lw":0.5}








rectangle = [min_lon-0.1, max_lon+0.1, min_lat-0.1, max_lat+0.1]
if region == "hula":
    rectangle[1] += 0.1
elif region == "kinneret":
    rectangle[1] += 0.1
    rectangle[3] += 0.1
# df_event = pd.read_csv(event_path,dtype={"evid":str})


coors = [min_lon, max_lon, min_lat, max_lat]
# origins = origins[(origins.lat>=coors[2]) &
#                     (origins.lat<=coors[3]) &
#                     (origins.lon>=coors[0]) &
#                     (origins.lon<=coors[1])]

station_df = []
for st in stations:
    temp = pd.read_csv(f"{dir_of_split_dat}/{st}.csv", dtype={"evid": str})
    temp["station"] = st
    station_df.append(temp)
station_df = pd.concat(station_df)

origins = pd.read_csv(event_path,
                   dtype={"evid": str})
station_df = station_df.merge(origins,how="left",left_on="evid",right_on="evid")

station_df = station_df[
    (station_df.lat>=coors[2]) &
    (station_df.lat<=coors[3]) &
    (station_df.lon>=coors[0]) &
    (station_df.lon<=coors[1]) &
    (station_df[f'Pp_{Type}']>=min_rec) #&
    # (np.abs(station_df.corr_coef)>=min_corr_coef)
]




sites = pd.read_csv(site_path)
df = station_df.merge(sites, how="left",on="station")

df.rename(
    {"lon": "ev_lon", "lat": "ev_lat", "longitude": "st_lon", "latitude": "st_lat"},
    inplace=True,
    axis=1
)
df["pos"] = df["station"].map(conditions)

df[["x","y"]] = pd.DataFrame(df["pos"].tolist(), index= df.index)
df["x"], df["y"] = df["x"]+df["st_lon"], df["y"]+df["st_lat"]
print(df.columns)



def polar_plot_st_position(df):

    fig = plt.figure(figsize=(10, 9))

    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(rectangle)

    unique_st = df.drop_duplicates("station")
    unique_ev = df.drop_duplicates("evid")

    ax.scatter(unique_st["st_lon"], unique_st["st_lat"], transform=ccrs.PlateCarree(), s=80, zorder=10, label='Station',
               color="yellow",
               marker='^', linewidths=0.5, edgecolors='k')

    fname = "/Users/guy.bendor/Natural Earth Map/map_files/ME_Shaded6.nc"
    fault_file = "/Users/guy.bendor/Natural Earth Map/map_files/SomeLayers4Guy/MainFaultsMS.shp"
    water_file = "/Users/guy.bendor/Natural Earth Map/map_files/SomeLayers4Guy/WaterBodiesSubset.shp"
    border_file = "/Users/guy.bendor/Natural Earth Map/map_files/SomeLayers4Guy/WorldBorders.shp"
    faults, water, da, border = get_map_layers(fault_file, water_file, fname, border_file)
    da = da.sel(x=slice(rectangle[0], rectangle[1]), y=slice(rectangle[3], rectangle[2]))
    ax.imshow(da.__xarray_dataarray_variable__.data[0], transform=ccrs.PlateCarree(), origin='upper',
              cmap="Greys_r", extent=rectangle, zorder=0, alpha=0.6)
    # da.__xarray_dataarray_variable__.plot(transform=ccrs.PlateCarree(),cmap="Greys_r", zorder=0, alpha=0.6)
    ax.add_feature(faults, facecolor='None', edgecolor='r', linestyle='--', zorder=2, label="Fault")
    ax.add_geometries(water, facecolor='lightcyan', edgecolor='black', linestyle='-',
                      linewidth=0.5, crs=ccrs.PlateCarree(), zorder=0)
    ax.add_feature(border, facecolor='None', zorder=1, edgecolor="k", linestyle=':',
                   linewidth=0.5)

    gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=0.5, color='black', alpha=0.3, linestyle='--', draw_labels=True)
    gl.top_labels = True
    gl.left_labels = True
    gl.right_labels = True
    gl.xlocator = mticker.FixedLocator(np.arange(np.floor(min_lon), max_lon + 2, 2))
    gl.ylocator = mticker.FixedLocator(np.arange(np.floor(min_lat), max_lat + 2, 2))
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    # print(origins.lon.values)
    # print(origins.lat.values)
    ax.scatter(unique_ev["ev_lon"], unique_ev["ev_lat"], c="r", s=50, alpha=0.5, linewidths=0.5, edgecolors='k',
               zorder=3, label="Event")
    ax.legend()

    unique_st = unique_st[["station","x","y",'st_lon','st_lat']].values

    for i in range(len(unique_st)):
        ax_sub = inset_axes(ax, width=1.5, height=1.5, loc=10,
                            bbox_to_anchor=(unique_st[i][1],unique_st[i][2]),
                            bbox_transform=ax.transData,
                            borderpad=0.0, axes_class=get_projection_class("polar"))
        ax_sub.patch.set_alpha(0.6)
        ax.annotate("",
                    xy=(unique_st[i][3],unique_st[i][4]), xycoords='data',
                    xytext=(unique_st[i][1],unique_st[i][2]), textcoords='data',
                    size=20, va="center", ha="center",
                    arrowprops=dict(arrowstyle="simple",
                                    connectionstyle="arc3,rad=-0.2"))

        strikes = df[df.station==unique_st[i][0]][f'angle_{Type}'].values
        bin_edges = np.arange(-5, 366, 10)
        number_of_strikes, bin_edges = np.histogram(strikes, bin_edges)
        number_of_strikes[0] += number_of_strikes[-1]
        half = np.sum(np.split(number_of_strikes[:-1], 2), 0)
        two_halves = np.concatenate([half, half])
        ax_sub.bar(np.deg2rad(bin_edges[:-2] + 5), two_halves,
               width=np.deg2rad(10), bottom=0.0, color='.8', edgecolor='k', lw=1)
        ax_sub.set_theta_zero_location('N')
        ax_sub.set_theta_direction(-1)
        ticks = np.arange(0, 360, 10)
        labels = np.where(ticks % 90==0, ticks, "")
        ax_sub.set_thetagrids(ticks, labels=labels,weight = 'bold')
        # ax_sub.set_rgrids(np.arange(1, two_halves.max() + 1, 2), angle=0, weight='black')
        if two_halves.max()>10 and two_halves.max()<=30:
            ax_sub.set_rgrids(np.arange(0, int(two_halves.max()/10)*10 + 5, 5), angle=0, weight='black')
        elif two_halves.max()>30 and two_halves.max()<=50:
            ax_sub.set_rgrids(np.arange(0, int(two_halves.max()/10)*10 + 10, 10), angle=0, weight='black')
        elif two_halves.max()>50:
            ax_sub.set_rgrids(np.arange(0, int(two_halves.max()/10)*10 + 10, 20), angle=0, weight='black')
        elif two_halves.max()>100:
            ax_sub.set_rgrids(np.arange(0, int(two_halves.max()/10)*10 + 10, 30), angle=0, weight='black')
        else:
            ax_sub.set_rgrids(np.arange(0, two_halves.max()+ 1, 2), angle=0, weight='black')


        # ax_sub.yaxis.set_major_locator(MultipleLocator(30))
        # ax_sub.yaxis.set_minor_locator(MultipleLocator(10))
        # ax_sub.yaxis.set_major_formatter('{x:.0f}°')  # +r'$^\circ$')
        # ax_sub.grid(b=True, which='both', axis="y", color='gray', linestyle='-', alpha=0.5)
        # ax_sub.xaxis.set_major_formatter(NullFormatter())
        ax_sub.set_title(unique_st[i][0], fontdict=dict(fontsize=8),
                         bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.5', alpha=0.4))



polar_plot_st_position(df)

plt.show()