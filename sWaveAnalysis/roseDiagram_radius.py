import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import os
import cartopy.crs as ccrs
# from fun_dir.polar_func import generate_map_view
from functions.baseFunctions import initial_bearing, haversine, convert_ang_to_xy, get_map_layers
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mcolors
import matplotlib as mpl
from cartopy import geodesic
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.path as mpath
import shapely
import cartopy.geodesic as gd
import xarray as xr
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from shapely.ops import unary_union
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker



# region = "all" # hula, kinneret
hula_stations = ["AMID","DSHN","GEM","HENS","KDZV","KSH0","LVBS","MTLA","RSPN"]
kinneret_stations = ["AMID","GLHS","HDNS","KNHM","MGDL","RMOT","TVR","RSPN"]

# stations = ["RSPN","AMID","GEM"]#,"AMID","GEM","NATI","MTLA"]#"GEM","MTLA","NATI"]
stations = ["RSPN","DSHN"]#,"AMID","GEM","MTLA","DSHN","LVBS","KDZV","HDNS","KNHM","KSH0","MGDL","RMOT","TVR","NATI","ENGV"]#,"GEM","NATI","MTLA"]

color_code = "corr_coef"


save_path = "figs"
data_dir = "splitParam"
pick_path = "../pickingEvents/pickFiles"

orig_df = pd.read_csv(
    "/Users/guy.bendor/PycharmProjects/ShearWave_splitting/data/main_data_files/relocated3.csv", dtype={"evid": str}
)
site_df = pd.read_csv("/Users/guy.bendor/PycharmProjects/ShearWave_splitting/data/main_data_files/sites.csv")
site_df.rename({'longitude': 'lon', 'latitude': 'lat', "station": 'name'}, axis=1, inplace=True)

# if region=="hula":
#     min_lat, max_lat = 32.9, 33.28
#     min_lon, max_lon = 35.52, 35.7
#     stations = hula_stations
# elif region == "kinneret":
#     min_lat, max_lat = 32.7, 32.9
#     min_lon, max_lon = 35.5, 35.65
#     stations = kinneret_stations
# elif region=="all":
#     min_lat, max_lat = 32.7, 33.28
#     min_lon, max_lon = 35.5, 35.7
# else:
#     raise ("region not defined")

# stations=["GEM"]
# min_lat, max_lat = 32.0, 34.0
# min_lon, max_lon = 33, 36

# rectangle = [min_lon-0.1, max_lon+0.2, min_lat-0.1, max_lat+0.1]

min_corr=0.9
min_rec=0.8
max_inc=90

min_corr=0.9
min_rec=0.8
max_inc=90
max_epi_dist = 40
confidence_level = 0
Type = "2d"
throw_cross_cor=True
throw_cross_cor=False
# Type = "3d"
# min_epi_dist = 10
# min_snr

# time1 = np.datetime64("2020-04-01")
# time2 = np.datetime64("2020-04-09")
# remove_swarm = False
#
# azi_cond = {
#     "HDNS": [[145, 165]],
#     "KDZV": [[135, 145]],
#     "DSHN": [[15, 25], [325, 345]],
#     "LVBS": [[135,145]],
#     "GEM": [[325,335]],
#     "RSPN": [[5,15], [335,345]]
#
#
# }

# df_event = pd.read_csv(event_path,dtype={"evid":str})

# coors = [min_lon, max_lon, min_lat, max_lat]

arrows = {'scale': 6.5, 'width': 0.002, 'headwidth': 1., 'headlength': 0.001, "scale_units": "inches","lw": 0.3}

fname = "/Users/guy.bendor/Natural Earth Map/map_files/ME_Shaded6.nc"
fault_file = "/Users/guy.bendor/Natural Earth Map/map_files/SomeLayers4Guy/MainFaultsMS.shp"
water_file = "/Users/guy.bendor/Natural Earth Map/map_files/SomeLayers4Guy/WaterBodiesSubset.shp"
border_file = "/Users/guy.bendor/Natural Earth Map/map_files/SomeLayers4Guy/WorldBorders.shp"
faults, water, da, border = get_map_layers(fault_file, water_file, fname, border_file)
colors = [
            'tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
            'tab:olive', 'tab:cyan'
        ]
def offset_calc(polar_ang, epi_direction):
    angs = np.array([polar_ang, epi_direction])
    dirs = np.c_[np.cos(np.radians(angs)), np.sin(np.radians(angs))]
    # x = np.cos(np.radians(angs))
    # y = np.sin(np.radians(angs))
    # offset = np.degrees(angle([x[0], y[0]], [x[1], y[1]]))
    # sign = 1 if ((0 <= angs[0] - angs[1] <= 180) or (angs[0] - angs[1] <= -180)) else -1

    # dirs = np.c_[x,y]
    offset = np.degrees(np.arccos(dirs[0] @ dirs[1]))
    offset = -offset if np.cross(dirs[0], dirs[1]) > 0 else offset
    return offset


def plot_against(df,x, y, z=None):
    global site_df,arrows
    # sta = site_df[site_df.name == st].iloc[0]
    # df["epi_direction"] = df.apply(lambda x: initial_bearing(sta.lon, sta.lat, x["lon"], x["lat"]), axis=1)
    # df["epi_distance"] = df.apply(lambda x: haversine(sta.lon, sta.lat, x["lon"], x["lat"]), axis=1)
    # df["offset"] = df.apply(lambda x: offset_calc(x["angles"], x["epi_direction"]), axis=1)
    df.loc[df.offset > 90, "offset"] -= 180
    df.loc[df.offset < -90, "offset"] += 180
    # df.loc[df.offset > 41, "offset"] -= 180

    angles = df["angles"].to_numpy() + 90

    # df["angles"] = np.where(
    #             df["angles"] >= 250,
    #             df["angles"] - 180,
    #             df["angles"]
    #         )

    temp = np.ones([2,len(angles)])
    temp[:, :] = angles, angles + 180
    temp = np.radians(temp)
    U = np.cos(temp)
    V = np.sin(temp)

    fig, ax = plt.subplots()

    if y == "offset" and x == "epi_direction" and not z:
        for u,v in zip(U,V):
            ax.quiver(df[x], df[y], u, v, ec="k", zorder=3, **arrows)
        ax.set_aspect('equal', adjustable='box')

    if z:
        ax.scatter(df[x], df[y], s=40, c=df[z], ec="k", lw=0.5)
    else:
        ax.scatter(df[x], df[y], s=20, c="r", ec="k", lw=0.5)


    ax.set_xlabel(x)
    ax.set_ylabel(y)
    plt.title(st)
    # print(xy.shape)
    # fig, ax = plt.subplots()
    #
    # ax.scatter(df[x], df[y])
    # plt.show()

def scatter_hist(x, y, ax_histx, ax_histy):
    # no labels
    ax_histx.tick_params(axis="x", labelbottom=False)
    ax_histy.tick_params(axis="y", labelleft=False)


    # now determine nice limits by hand:
    binwidth = 10
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax/binwidth) + 1) * binwidth

    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histx.hist(x, bins=bins, edgecolor='black', linewidth=1.2)
    ax_histy.hist(y, bins=bins, orientation='horizontal', edgecolor='black', linewidth=1.2)

def generate_map_view(fig, ax, ev, sta, **kwargs):
    """
    Generates a map view containing the station and all specified events.
    :param fig: matplotlib fig type object
    :param ax: ax of the fig type object
    :param ev: a dictionary containing events info.
                Format: {"lon": list(), "lat": list(), "evid": list(),
                         "angle": list() - optional, "rule": list() - optional}
    :param sta: a dictionary containing station info.
                Format: {"lon": val, "lat": val, "name": val,
                         "angle": list() - optional, "rule": list() - optional}
    :param kwargs:
                arrows: quiver arrows shape.
                        Default is: {'length': 8*1e-5, 'headlength': 4, 'headwidth': 3, 'width': 0.004}
                color_rules: used to establish rules for arrow color coding.
                        Format: {"name": name of rule (tw, Ps, offset...), "min": val, "max": val}
                resolution: default is 8 (higher values take longer to plot).
                twosided: default is False. Whether or not use two sided quiver arrows.
                on_event: default is True. whether or not to plot the quiver arrows on the event or station location.
                cmap: color code design for quiver arrows.
                add_circle: Radius #True/False
    :return:
    """
    arrow_dict = {"scale_units":'inches'}#'scale': 4.,
                  #'width': 0.01,
                  #'headwidth':4,
                  #'headlength': 2,
                  #'minshaft': 4}

    if "arrows" in kwargs:
        arrow_dict.update(kwargs["arrows"])

    on_event = kwargs["on_event"] if "on_event" in kwargs else True
    twosided = kwargs["twosided"] if "twosided" in kwargs else False

    if len(ev["evid"]) != 1 and len(sta["name"]) == 1:
        center = [sta["lon"][0], sta["lat"][0]]
    elif len(ev["evid"]) == 1 and len(sta["name"]) != 1:
        center = [ev["lon"][0], ev["lat"][0]]
    else:
        center = [np.average(ev["lon"]+sta["lon"]), np.average(ev["lat"]+sta["lat"])]

    if "angle" in ev.keys() and on_event:
        arrow_position = [ev["lon"], ev["lat"]]# if on_event == True else [[sta["lon"][0]] * len(ev["angle"]),
                                                #                      [sta["lat"][0]] * len(ev["angle"])]
    else:
        arrow_position = [sta["lon"], sta["lat"]]

    arrow_position = np.array(arrow_position)




    if "custum_size" in kwargs:
        coordinates = kwargs["custum_size"]
    else:
        r = 100000
        lim_coords = geodesic.Geodesic().direct(center, [180, 270, 0, 90], [r + 50000] * 4)
        coordinates = [lim_coords[1, 0], lim_coords[3, 0], lim_coords[0, 1], lim_coords[2, 1]]
    ax.set_extent(coordinates, crs=ccrs.PlateCarree())

    ax.scatter([sta["lon"]], [sta["lat"]],
               transform=ccrs.PlateCarree(), s=100, zorder=4, label='scatter', color='yellow', marker='^',
               linewidths=0.5, edgecolors='k')

    ax.scatter(ev["lon"], ev["lat"],
               transform=ccrs.PlateCarree(), s=10, zorder=4, label='scatter', color='k', marker='.',
               linewidths=0.5, edgecolors='k'
               )

    if len(ev["evid"]) == 1:
        plt.suptitle(f"{ev['evid'][0]}")
        for i in range(len(sta["name"])):
            ax.text(sta["lon"][i], sta["lat"][i] + 0.01, sta["name"][i], transform=ccrs.PlateCarree())
    else:
        plt.suptitle(f"{sta['name'][0]}")

    if "angle" in ev.keys():
        angles = ev["angle"]
        if "rule" in ev.keys():
            c = ev["rule"]

    elif "angle" in sta.keys():
        angles = sta["angle"]
        if "rule" in sta.keys():
            c = sta["rule"]
    else:
        return fig, ax

    xy = convert_ang_to_xy(np.array(angles), 1)

    if any([i not in kwargs for i in ["color_rules", "cmap"]]):
        norm = None
        cmap = None
        c = np.ones(len(angles))
        add_cbar = False
    else:
        norm = mcolors.Normalize(vmin=kwargs["color_rules"]["min"],
                                           vmax=kwargs["color_rules"]["max"], clip=False)
        cmap = kwargs["cmap"]
        add_cbar = True


    if twosided:
        pass
    else:
        if on_event:
            xy = np.delete(xy, 0, axis=0)
        else:
            xy = np.delete(xy, 1, axis=0)
    for i in range(xy.shape[0]):
        ax.quiver(arrow_position[0], arrow_position[1], xy[i, 0, :], xy[i, 1, :], c,
                  transform=ccrs.PlateCarree(),**arrow_dict, angles='xy',cmap=cmap, norm=norm, ec='k',zorder=3)
                  # headwidth=arrow_dict['headwidth'], angles='xy', scale_units='xy', scale=arrow_dict["length"],
                  # headlength=arrow_dict['headlength'], width=arrow_dict['width'], minshaft=arrow_dict['minshaft'],
                  # cmap=cmap, norm=norm, ec='k', lw=1,zorder=2)
    # print(kwargs)
    if "add_circle" in kwargs and isinstance(kwargs["add_circle"],list):
        # Adds a circle of radii 'r' - for my purposes it is set to 100 km
        for radii in kwargs["add_circle"]:
            circle_points = geodesic.Geodesic().circle(
                lon=center[0], lat=center[1], radius=radii, n_samples=200, endpoint=False
            )
            geom = shapely.geometry.Polygon(circle_points)
            ax.add_geometries((geom,), crs=ccrs.PlateCarree(), facecolor='none', edgecolor='k', linewidth=1, zorder=2)

    if add_cbar:


        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        if "dates" in kwargs and kwargs["dates"]:
            loc = mdates.AutoDateLocator()
            cbar = plt.colorbar(sm, ticks=loc, format=mdates.AutoDateFormatter(loc), ax=ax, pad=0.1)
        else:
            cbar = plt.colorbar(sm, ax=ax, pad=0.1)
        cbar.set_label(kwargs["color_rules"]["name"], y=0.5, rotation=90, fontsize=16)

    if "xrDa" in kwargs:
        xrDa = kwargs["xrDa"]
        da = xrDa.sel(x=slice(coordinates[0], coordinates[1]), y=slice(coordinates[3], coordinates[2]))
        ax.imshow(da.__xarray_dataarray_variable__.data[0], transform=ccrs.PlateCarree(), origin='upper',
                  cmap="Greys_r", extent=coordinates, zorder=0, alpha=0.6)

    if "faults" in kwargs:
        faults = kwargs["faults"]
        ax.add_feature(faults, facecolor='None', edgecolor='r', linestyle='--', zorder=2)

    if "water" in kwargs:
        water = kwargs["water"]
        ax.add_geometries(water, facecolor='lightcyan', edgecolor='black', linestyle='-',
                          linewidth=0.5, crs=ccrs.PlateCarree(), zorder=1)

    if "border" in kwargs:
        border = kwargs["border"]
        ax.add_feature(border, facecolor='None', zorder=1, edgecolor="k", linestyle=':',
                       linewidth=0.5)

    gl = ax.gridlines(crs=ccrs.PlateCarree(), linewidth=0.5, color='black', alpha=0.3, linestyle='--', draw_labels=True)
    gl.top_labels = True
    gl.left_labels = True
    gl.right_labels = True
    gl.xlocator = mticker.FixedLocator(np.arange(0, 50, 1))
    gl.ylocator = mticker.FixedLocator(np.arange(0, 50, 1))
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    return fig, ax
# def test(df,sta,x,y,z):
#     global site_df, arrows
#     sta = site_df[site_df.name == st].iloc[0]
#     df["epi_direction"] = df.apply(lambda x: initial_bearing(sta.lon, sta.lat, x["lon"], x["lat"]), axis=1)
#     df["epi_distance"] = df.apply(lambda x: haversine(sta.lon, sta.lat, x["lon"], x["lat"]), axis=1)
#     df["offset"] = df.apply(lambda x: offset_calc(x["angles"], x["epi_direction"]), axis=1)
#     df.loc[df.offset > 90, "offset"] -= 180
#     df.loc[df.offset < -90, "offset"] += 180
#     df["angles"] = np.where(
#         df["angles"] >= 180,
#         df["angles"] - 180,
#         df["angles"]
#     )
#
#     df.sort_values("epi_direction",inplace=True,ignore_index=True)
#
#
#     fig, ax = plt.subplots()
#
#     ax.scatter(df[x], df[y],c=df[z])
#     # ax.scatter(df["epi_direction"], df["angles"]-180)
#     # ax.plot(x_av,y_av)
#     plt.title(st)
#     plt.show()
def plot_x_y(df,x_label,y_label,title,z_label=None,zstep=None, maxZval=None, minZval=0):
    # print("zstep",zstep)
    # print(df[[x_label,y_label]].to_numpy().T)
    if z_label:
        x,y,z = df[[x_label,y_label,z_label]].to_numpy().T
    else:
        x, y = df[[x_label, y_label]].to_numpy().T
    binwidth = 10
    bins = np.arange(0, 180 + binwidth, binwidth)

    R = np.arange(0, 361, 10)
    # arr = np.zeros([R.shape[0] - 1, 2])
    # test_arr = np.zeros([R.shape[0] - 1, 2])
    test_arr = []
    for i in range(0, len(R) - 1):

        # test_arr[i, 0] = R[i:i + 2].mean()
        temp = y[(x >= R[i]) & (x < R[i + 1])]
        if len(temp) != 0:
            xAng, yAng = np.cos(np.radians(temp)), np.sin(np.radians(temp))
            ang = np.degrees(np.arctan2(yAng.sum(),xAng.sum()))%360
            if ang>=180: ang-=180
            test_arr.append([R[i:i + 2].mean(),ang])

            # arr[i, 1] = temp.mean()
            # test_arr[i, 1] = np.degrees(
            #     np.arctan2(np.sin(np.radians(y[(x >= R[i]) & (x < R[i + 1])])).sum(),
            #                np.cos(np.radians(y[(x >= R[i]) & (x < R[i + 1])])).sum())
            # )%180
    test_arr = np.array(test_arr)
    if x_label=="angles":
        x[x>=180] = x[x>=180]-180
        res = np.histogram(x, bins=bins)
        x[x>=bins[res[0].argmin()]]-=180

    if y_label=="angles":
        y[y>=180] = y[y>=180]-180
        res = np.histogram(y, bins=bins)
        # print(y[y>=bins[res[0].argmin()]])
        y[y>=bins[res[0].argmin()]]-=180
        # print(test_arr[:,1]>=bins[res[0].argmin()])
        test_arr[test_arr[:,1]>=bins[res[0].argmin()], 1]-=180

    bins[bins>=bins[res[0].argmin()]]-=180

    # start with a square Figure
    fig,ax = plt.subplots()



    divider = make_axes_locatable(ax)
    ax_histy = divider.append_axes("right", 0.6, pad=0.1, sharey=ax)
    ax_histx = divider.append_axes("top", 0.6, pad=0.1, sharex=ax)
    cax = divider.append_axes("right", 0.1, pad=0.1)
    ax_histy.yaxis.set_tick_params(labelleft=False)
    ax_histx.xaxis.set_tick_params(labelbottom=False)
    ax_histx.set_title(title)

    # the scatter plot:
    if z_label:
        if z_label == "confidence":
            multi = [y[np.where(z == val)[0]] for val in range(minZval, maxZval+zstep, zstep)]
            multi_x = [x[np.where(z == val)[0]] for val in range(minZval, maxZval+zstep, zstep)]
        else:
            multi = [y[np.where((z >= dist) & (z < dist + zstep))[0]] for dist in range(minZval, maxZval, zstep)]
            multi_x = [x[np.where((z >= dist) & (z < dist + zstep))[0]] for dist in range(minZval, maxZval, zstep)]
        # colors = ['tab:blue', 'tab:orange', 'tab:green','tab:purple','tab:red']

        bounds = np.arange(minZval, maxZval+zstep/2, zstep)
        # print(len(bounds))
        # if len(bounds)>4:
        #     colors+=['tab:red']
        cmap = (mcolors.ListedColormap(colors[:len(multi)]))

        norm = mcolors.BoundaryNorm(bounds, cmap.N)
        for i in range(len(multi)):
            sct = ax.scatter(multi_x[i], multi[i], c=colors[i], s=20,alpha=0.7,edgecolor="k",lw=0.5)
        cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),cax=cax,label="Depth [km]")
        # cbar.set_label('Depth', rotation=270)
        # cbar = plt.colorbar(sct,cax=cax)
        # cbar.set_label(z_label)
    else:
        ax.scatter(x, y, c="k", s=20)
    # ax.scatter(arr.T[0], arr.T[1])
    # ax.scatter(test_arr.T[0], test_arr.T[1])

    # ax = fig.add_axes(rect_scatter)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel("epi direction")
    ax.set_ylabel("init polarization")

    # ax.scatter(x, y, c="k", s=20)

    if x_label == "angles":
        ax.set_xlim(min(bins), max(bins))
        ax.set_ylim(0, 360)
    if y_label == "angles":
        ax.set_ylim(min(bins), max(bins))
        ax.set_xlim(0, 360)

    # ax_histx.hist(x, bins=bins, edgecolor='black', linewidth=1.2)
    # ax_histy.hist(y, bins=bins, orientation='horizontal', edgecolor='black', linewidth=1.2)
    # ax_histx = fig.add_axes(rect_histx, sharex=ax)
    # ax_histy = fig.add_axes(ax_histy, sharey=ax)

    # use the previously defined function
    # now determine nice limits by hand:
    binwidth = 10
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax / binwidth) + 1) * binwidth

    bins = np.arange(-lim, lim + binwidth, binwidth)
    ax_histx.hist(x, bins=bins, edgecolor='black', linewidth=1.2)
    if z_label:
        ax_histy.hist(
            multi, bins=bins, orientation='horizontal', edgecolor='black',
            linewidth=0.2, stacked=True
        )
        # for hi in multi:
        #     ax_histy.hist(hi,bins=bins, orientation='horizontal', edgecolor='black',
        #     linewidth=0.2,alpha=0.3)

    else:
        ax_histy.hist(y, bins=bins, edgecolor='black', linewidth=1.2)
    return fig


def plot_time_vs_pol(df, name, delay_color=None):

    # ["delay", "hypo_dist", "epi_dist", "depth", "epi_dir"]
    df = df.sort_values(by=["origin_times"], ignore_index=True)

    strikes = df.angles.values
    times = df.origin_times.values

    bin_edges = np.arange(-5, 366, 10)
    number_of_strikes, bin_edges = np.histogram(strikes, bin_edges)
    number_of_strikes[0] += number_of_strikes[-1]
    half = np.sum(np.split(number_of_strikes[:-1], 2), 0)
    two_halves = np.concatenate([half, half])

    fig = plt.figure(figsize=(16, 8))
    # ax = fig.add_subplot(211, projection='polar')
    ax = plt.axes([0., 0.5, 0.4, 0.4], projection='polar')

    ax.bar(np.deg2rad(bin_edges[:-2] + 5), two_halves,
           width=np.deg2rad(10), bottom=0.0, color='.8', edgecolor='k', lw=1)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.arange(0, 360, 10), labels=np.arange(0, 360, 10))
    ax.set_rgrids(np.arange(1, two_halves.max() + 1, 2), angle=0, weight='black')

    ax1 = plt.axes([0.05, 0.08, 0.35, 0.37])  # fig.add_subplot(212)

    if delay_color:
        s = ax1.scatter(times, strikes, s=20, c=df[delay_color].values, edgecolor="k", lw=0.5, alpha=0.7)
        divider = make_axes_locatable(ax1)
        # cax = divider.append_axes('right', size='5%', pad=0.05)
        # fig.colorbar(s, cax=cax, orientation='vertical')
        fig.colorbar(s, ax=ax1, label=delay_color)
    else:
        s = ax1.scatter(times, strikes, s=20, edgecolor="k", lw=0.5, alpha=0.7)


    ax_histy = divider.append_axes("right", 0.6, pad=0.1, sharey=ax1)
    ax_histy.yaxis.set_tick_params(labelleft=False)
    binwidth = 10
    bins = np.arange(180, 360 + binwidth, binwidth)
    ax_histy.hist(strikes, bins=bins, orientation='horizontal')

    ax1.set_xlabel("Date", fontsize=15)
    plt.setp(ax1.get_xticklabels(), ha="right", rotation=45)
    ax1.set_ylabel(r"$\Phi [^\circ]$", fontsize=15)

    ax2 = plt.axes([0.4, 0.1, 0.45, 0.8], projection=ccrs.PlateCarree())

    sta = site_df[site_df.name == name].to_dict(orient="list")

    df.rename({'event': 'evid'}, axis=1, inplace=True)

    baz = df.copy()

    baz.rename({'angles': 'angle', "delay": "rule"}, axis=1, inplace=True)

    ev = {i: baz[i].values for i in ["evid", "lon", "lat", "angle", "rule"]}

    generate_map_view(fig, ax2, ev, sta,
                      twosided=True, arrows=arrows, add_circle=False, dates=False, custum_size=rectangle)

def make_rose_diagram_and_delay_analysis(df, name, azi_cond=None,num_samp_av=None, delay_color=None, R=None):

    # df = df[[
    #     "evid", "angles", "origin_times", "delay", 'epi_direction', 'epi_distance', 'offset', 'depth', 'hypo_dist',"corr_coef"
    # ]].copy()

    df = df.sort_values(by=["origin_times"],ignore_index=True)

    strikes = df.angles.values
    times = df.origin_times.values
    delayes = df.delay.values
    print(len(delayes))
    distances = df.epi_dist.values
    epi_dir = df.epi_dir.values
    corr_coef = df.corr_coef.values
    if delay_color:
        delay_color_val = df[delay_color].values
    # print(distances)
    # print(epi_dir)
    # hypo_dists = df.hypo_dist.values

    # epi_distance = df.epi_distance.values
    # depths = df.depth.values
    # df["corr_coef"] = np.abs(df.corr_coef.values)

    # ["hypo_dist","epi_dist","epi_dir","depth","corr_coef"]
    inx = np.where(corr_coef>min_corr)[0]
    times = times[inx]
    delayes = delayes[inx]
    delay_color_val = delay_color_val[inx]
    # print(len(delayes))
    # print(len(delayes)-num_samp_av+1)
    if num_samp_av:
        av_delays = np.zeros(len(delayes)-num_samp_av+1)
        av_times = np.zeros(len(delayes)-num_samp_av+1, dtype="<M8[ns]")
        # times = df.origin_times.values
        td = np.timedelta64(1, "M")
        yM = times.astype('datetime64[M]')
        # yM1 = yM.copy()+td
        # print(yM)
        # print(yM1)


        # creates an array of indices, sorted by unique element
        idx_sort = np.argsort(yM)

        # sorts records array so all unique elements are together
        sorted_records_array = yM[idx_sort]

        # returns the unique values, the index of the first occurrence of a value, and the count for each element
        vals, idx_start, count = np.unique(sorted_records_array, return_counts=True, return_index=True)

        # splits the indices into separate arrays
        res = np.split(idx_sort, idx_start[1:])

        # filter them with respect to their size, keeping only items occurring more than once
        minNumMonth = 3
        vals = vals[count >= minNumMonth]
        res = filter(lambda x: x.size >= minNumMonth, res)
        av_times = vals
        av_delays = []#np.zeros(len(av_times))
        for inx in res:
            # av_delays.append(np.median(delayes[inx]))
            av_delays.append(delayes[inx].mean())



        # print(vals)
        # print(list(res))
        #
        # f_times = times.copy().astype("float64")
        #
        # for i in range(len(delayes)-num_samp_av+1):
        #     av_delays[i] = delayes[i:i + num_samp_av].mean()
        #     av_times[i] = f_times[i:i + num_samp_av].mean().astype("<M8[ns]")

    # step = 10
    # multi_two_halves = []
    # multi_binEdges = []
    # for dirct in range(0, 360, step):
    #     temp = strikes[np.where((dirct <= epi_dir) & (epi_dir < dirct + step))[0]]
    #     number_of_strikes, binEdges = np.histogram(temp, bin_edges)
    #     number_of_strikes[0] += number_of_strikes[-1]
    #     half = np.sum(np.split(number_of_strikes[:-1], 2), 0)
    #     two_halves = np.concatenate([half, half])
    #     multi_two_halves.append(two_halves)
    #     multi_binEdges.append(np.deg2rad(binEdges[:-2] + 5))
    # for i in range(len(multi_two_halves)):
    #     bot = multi_two_halves[i - 1] if i != 0 else 0
    #     ax.bar(multi_binEdges[i], multi_two_halves[i],
    #            width=np.deg2rad(10), bottom=bot, edgecolor='k', lw=1)
    bin_edges = np.arange(-5, 366, 10)
    number_of_strikes, bin_edges = np.histogram(strikes, bin_edges)
    number_of_strikes[0] += number_of_strikes[-1]
    half = np.sum(np.split(number_of_strikes[:-1], 2), 0)
    two_halves = np.concatenate([half, half])

    fig = plt.figure(figsize=(16, 8))
    # ax = fig.add_subplot(211, projection='polar')
    ax = plt.axes([0.,0.5,0.4,0.4],projection='polar')
    ax.bar(np.deg2rad(bin_edges[:-2]+5), two_halves,
           width=np.deg2rad(10), bottom=0.0, color='.8', edgecolor='k', lw=1)
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.arange(0, 360, 10), labels=np.arange(0, 360, 10))
    ax.set_rgrids(np.arange(1, two_halves.max() + 1, 2), angle=0, weight='black')

    ax1 = plt.axes([0.05,0.08,0.35,0.37])#fig.add_subplot(212)

    if azi_cond:
        # print(delay_color)
        inx_in = np.array([], dtype=int)
        #print(len(strikes),len(times),len(delayes))
        for az_c in azi_cond:
            I = np.where(
                ((strikes >= az_c[0]) & (strikes <= az_c[1])) |
                ((strikes >= (az_c[0] + 180) % 360) & (strikes <= (az_c[1] + 180) % 360))
            )[0]
            #print(I)
            ax1.scatter(times[I], delayes[I], label=f"between {az_c}", zorder=1)
            inx_in = np.append(inx_in,I)

        out_t = np.delete(times, inx_in)
        out_d = np.delete(delayes, inx_in)
        ax1.scatter(out_t, out_d, label="outside", zorder=0)

    else:
        # print(delay_color)
        if delay_color:

            s = ax1.scatter(times,delayes,s=20,c=delay_color_val,edgecolor="k",lw=0.5,alpha=0.7)

            # cax = divider.append_axes('right', size='5%', pad=0.05)
            # fig.colorbar(s, cax=cax, orientation='vertical')
            fig.colorbar(s, ax=ax1,label="corr coef")
        else:
            ax1.scatter(times, delayes, s=10, c="k")
        if num_samp_av:
            ax1.plot(av_times,av_delays,"orange",ls="--",label=f"moving average per month",zorder=0,alpha=0.3)
            # ax1.axhline(y=delayes.mean(),c="orange",ls="--",label="Mean time delay")
            ax1.legend()
    ax1.set_xlabel("Date",fontsize=15)
    plt.setp(ax1.get_xticklabels(), ha="right", rotation=45)
    ax1.set_ylabel(r"$\Delta t$ [s]",fontsize=15)

    # print(min(delayes),max(delayes))
    divider = make_axes_locatable(ax1)
    ax_histy = divider.append_axes("right", 0.6, pad=0.1, sharey=ax1)
    ax_histy.yaxis.set_tick_params(labelleft=False)
    binwidth = 0.005
    # print(delayes)
    if delayes.size>0:
        bins = np.arange(0, max(delayes) + binwidth, binwidth)
        ax_histy.hist(delayes, bins=bins, orientation='horizontal')

    ax2 = plt.axes([0.4,0.1,0.45,0.8],projection=ccrs.PlateCarree())

    sta = site_df[site_df.name == name].to_dict(orient="list")

    df.rename({'event': 'evid'}, axis=1, inplace=True)

    # print(df.columns)
    # print(orig_df.columns)
    # baz = df.merge(orig_df, on="evid")
    # print(baz.columns)
    baz = df.copy()


    baz.rename({'angles': 'angle', "delay": "rule"}, axis=1, inplace=True)

    ev = {i: baz[i].values for i in ["evid", "lon", "lat", "angle", "rule"]}

    generate_map_view(fig, ax2, ev, sta,
                      twosided=True, arrows=arrows, add_circle=R,
                      dates=False, custum_size=rectangle,
                      xrDa=da,faults=faults,water=water,border=border)

    #fig.suptitle(f'Rose Diagram for fast S-wave polarization directions\n[{name}]', y=0.98, fontsize=15)

    from string import ascii_lowercase
    # for i,axis in enumerate([ax,ax1,ax2]):
    #     axis.annotate(f'1.{ascii_lowercase[i]}', xy=(0, 0), xytext=(0.94, 0.03), xycoords='axes fraction',
    #                      textcoords='axes fraction',bbox=dict(facecolor='white', edgecolor='black'))
    # ax.annotate(f'4.{ascii_lowercase[0]}', xy=(0, 0), xytext=(1.1, -0.05), xycoords='axes fraction',
    #               textcoords='axes fraction', bbox=dict(facecolor='white', edgecolor='black'))
    #
    # ax1.annotate(f'4.{ascii_lowercase[1]}', xy=(0, 0), xytext=(0.95, 0.05), xycoords='axes fraction',
    #             textcoords='axes fraction', bbox=dict(facecolor='white', edgecolor='black'))
    #
    # ax2.annotate(f'4.{ascii_lowercase[2]}', xy=(0, 0), xytext=(0.9, 0.05), xycoords='axes fraction',
    #              textcoords='axes fraction', bbox=dict(facecolor='white', edgecolor='black'))

    # fig.tight_layout(pad=0.7)#w_pad=1.01,h_pad=1.01)
    fig.subplots_adjust(right=0.8)

    return fig





for st in stations:
    print(st)
    df = pd.read_csv(f"{data_dir}/{st}.csv",dtype={"evid": str})
    pick_df = pd.read_csv(
        f"{pick_path}/{st}.csv", dtype={"evid": str}
    )
    pick_df = pick_df[pick_df.Phase == "S"]
    df = df.merge(pick_df[['evid', 'confidence']], on="evid")
    # df = df[]
    # # df = df[df.rect_after_delay > 0.7]
    # df = df[]
    # print(df.columns)
    # df.rename({"rotation_3d": "angles","event": "evid"},axis=1,inplace=True)

    df.rename({f"angle_{Type}": "angles","time_delay":"delay"}, axis=1, inplace=True)
    if Type == "3d":
        df = df[df['Ps_3d']>=min_rec]
    elif Type=="2d":
        df = df[df.Pp_2d >= min_rec]
    # df = df[df.evid=="123806"]
    df.dropna(subset=["confidence"], inplace=True)
    df = df[df.confidence <= confidence_level]
    df = df.merge(orig_df, on="evid")
    sta = site_df[site_df.name == st].iloc[0]
    df["epi_dir"] = df.apply(lambda x: initial_bearing(sta.lon, sta.lat, x["lon"], x["lat"]), axis=1)
    df["epi_dist"] = df.apply(lambda x: haversine(sta.lon, sta.lat, x["lon"], x["lat"]), axis=1)
    df["offset"] = df.apply(lambda x: offset_calc(x["angles"], x["epi_dir"]), axis=1)
    df["hypo_dist"] = df[['epi_dist', 'depth']].apply(lambda x: np.sqrt(x[0] ** 2 + x[1] ** 2), axis=1)
    df.corr_coef = df.corr_coef.apply(np.abs)
    # df = df[(df.lat >= coors[2]) &
    #         (df.lat <= coors[3]) &
    #         (df.lon >= coors[0]) &
    #         (df.lon <= coors[1]) &
    #         (df.rect_of_first_samples >= min_rec) &
    #         (np.abs(df.corr_coef) >= min_corr) &
    #         (np.abs(df.inclination_3d)<=max_inc)
    # ]

    df = df[(df.epi_dist <= max_epi_dist) &
            (np.abs(df.Inclination_3d) <= max_inc)
            ]

    if throw_cross_cor:
        df = df[df.corr_coef>=min_corr]
    # coors = [min_lon, max_lon, min_lat, max_lat]
    d_ang = (max_epi_dist+20)*360/40000
    rectangle = [sta.lon-d_ang,sta.lon+d_ang,sta.lat-d_ang,sta.lat+d_ang]

    df["UTCtime"] = df["UTCtime"].astype("datetime64[ms]")
    df.rename({"UTCtime": "origin_times"}, axis=1, inplace=True)

    if len(df) == 0:
        continue
    print(st)
    circles = [max_epi_dist*1000]
    if "min_epi_dist" in globals():
        circles.append(min_epi_dist*1000)
        df = df[(df.epi_dist >= min_epi_dist)]
    # print(df.columns)

    fig = make_rose_diagram_and_delay_analysis(df, st, delay_color=color_code,R=circles)#,num_samp_av=5)

    # fig2 = plot_x_y(df, "epi_dir", "angles",st,z_label="confidence",zstep=1,maxZval=2)#z_label="epi_dist",zstep=10,maxZval=max_epi_dist)
    fig2 = plot_x_y(df, "epi_dir", "angles",st,z_label="epi_dist",zstep=10,maxZval=max_epi_dist)
    # fig2 = plot_x_y(df, "epi_dir", "angles", st, z_label="epi_dir", zstep=30, maxZval=180,minZval=150)

    if not os.path.exists(save_path):
        os.mkdir(save_path)
    fig.savefig(f"{save_path}/{st}_map")
    fig2.savefig(f"{save_path}/{st}_graph")
    # plot_time_vs_pol(df, st,delay_color="delay")

    # plot_against(df, st, "epi_direction", "offset")
    # scatter_hist(df, st, "epi_direction", "angles", "inclination_3d")
    # scatter_hist(df, st, "epi_direction", "offset")
    # test(df, st, "epi_direction","offset", "origin_times","epi_distance")




    # print(df)
    # angles = df.dir_of_first_samples.values
    # generate_rose_diagram(angles,st)

plt.show()
