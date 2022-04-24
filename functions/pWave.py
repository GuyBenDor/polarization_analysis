import obspy
import pandas as pd
import numpy as np
import os
import matplotlib
from functions.baseFunctions import trace_data, find_nearest,initial_bearing,haversine,wavelength,arrow_dic\
    ,convert_ang_to_xy,offset_calc,mat_rotation

# from fun_dir.polar_func import angle, initial_bearing, haversine, wavelength, convert_ang_to_xy
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
# from fun_dir.slider_func import Arrow3D, arrow_dic
import matplotlib.patches as mpatches
import mpl_toolkits.mplot3d.art3d as art3d

cm = plt.cm.get_cmap('Greys')
normalize = matplotlib.colors.Normalize(vmin=0, vmax=1)



# def offset_calc(vec,epi_direction):
#     epi = np.radians(epi_direction)
#     epi_vec = np.array([np.sin(epi),np.cos(epi)])
#     epi_vec = epi_vec/np.sqrt(np.sum(epi_vec**2))
#     vec = vec[:2]/np.sqrt(np.sum(vec[:2]**2))
#
#     offset = np.degrees(angle(vec, epi_vec))
#     sign = 1 if ((0 <= vec[0] - epi_vec[1] <= 180) or (vec[0] - epi_vec[1] <= -180)) else -1
#     return sign * offset

# def plot_3d_motion(st, data):
#     fig = plt.figure(figsize=(6, 5))
#     ax = plt.axes(projection='3d')
#     xn, yn, zn = st.data[:3, st.noise_inds[0]:st.wave_inds[-1]]
#     x, y, z = st.data[:3, st.wave_inds]
#     t = range(len(z))
#
#     mx = np.sqrt((st.data[:3, st.noise_inds[0]:st.wave_inds[-1]]**2).sum(axis=0)).max()
#
#     # ax.plot3D(xn, yn, zn, 'gray', zorder=0,lw=0.5)
#     ax.plot3D(x, y, z, 'gray', zorder=0, lw=0.5)
#     sct = ax.scatter3D(x, y, z, c=t, cmap=cm, zorder=1)
#
#     pca_vec = np.c_[np.ravel(st.dic["components"])]
#     #pca_vec /= np.sqrt((pca_vec**2).sum())
#     pca_vec = np.concatenate((pca_vec, np.zeros(pca_vec.shape)), axis=1)
#     V0 = pca_vec[:3]*mx  # * self.ev.pca_dic["variance"][0]
#     ax.plot3D(V0[0], V0[1], V0[2],"r")
#
#     hor_proj = np.zeros(V0.shape)
#     hor_proj[:-1, :] = V0[:-1, :].copy()
#     hor_proj = hor_proj# * (hor_proj ** 2).sum() ** 0.5
#     ax.plot3D(hor_proj[0], hor_proj[1], hor_proj[2], "r--")
#
#     epi = convert_ang_to_xy(data.epi_direction, 1)
#     epi = epi[0]
#     epi = np.c_[np.array([np.ravel(epi)]), 0]
#     epi = (epi /(epi ** 2).sum() ** 0.5)* (hor_proj ** 2).sum() ** 0.5  # )
#     epi = np.append(epi, np.zeros(epi.shape), axis=0).T
#     ax.plot3D(epi[0], epi[1], epi[2], "g")
#
#     # for key, val in lines.items():
#     #     val["array"] = val["array"] * mx
#     #     ax.plot3D(val["array"][0], val["array"][1], val["array"][2], c=val["color"], ls=val["style"], label=key)

def plot_2d_motions(st,data):
    fig = plt.figure(constrained_layout=False, figsize=(12, 6))
    gs1 = fig.add_gridspec(nrows=1, ncols=2)#, left=0.05, right=0.48, wspace=0.05)
    # ax = fig.add_subplot(gs1[:, 0],projection="3d")
    ax0 = fig.add_subplot(gs1[:, 0])
    ax1 = fig.add_subplot(gs1[:, 1])
    # print(st.__dict__.keys())
    # print(st.dic.keys())
    xn, yn, zn, deltimesn, timesn = st.data[:, st.noise_inds[0]:st.wave_inds[-1]]
    x, y, z, deltimesn_, times_ = st.data[:, st.wave_inds]
    # print(st.dic["angle"])
    xtag, ytag, ztag = mat_rotation(st.dic["angle"],st.data[:3, st.wave_inds])


    times = st.data[
            4,
            st.wave_inds[0] - int(st.stats.sampling_rate * 0.5):st.wave_inds[-1] + int(st.stats.sampling_rate * 0.25)
            ].astype("datetime64[ms]")

    delta_times = st.data[
                  3,
                  st.wave_inds[0] - int(st.stats.sampling_rate * 0.5):st.wave_inds[-1] + int(
                      st.stats.sampling_rate * 0.25)
                  ]
    # print(st.stats)
    t = range(len(z))

    mx = np.sqrt((st.data[:3, st.noise_inds[0]:st.wave_inds[-1]] ** 2).sum(axis=0)).max()
    # print(x)
    # print(xtag)
    # ax.plot3D(x, y, z, 'gray', zorder=0,lw=0.5)
    # sct = ax.scatter3D(x, y, z, c=t, cmap=cm, zorder=1)
    ax0.plot(x-x.mean(), y-y.mean(), 'gray', zorder=1, lw=0.5)
    sct0 = ax0.scatter(x-x.mean(), y-y.mean(), c=t, cmap=cm, zorder=2)
    ax1.plot(ytag-ytag.mean(), ztag-ztag.mean(), 'gray', zorder=1, lw=0.5)
    sct1 = ax1.scatter(ytag-ytag.mean(), ztag-ztag.mean(), c=t, cmap=cm, zorder=2)

    pca_vec = np.c_[np.ravel(st.dic["components"])]
    # pca_vec /= np.sqrt((pca_vec**2).sum())
    pca_vec = np.concatenate((pca_vec, np.zeros(pca_vec.shape)), axis=1)
    V0 = pca_vec[:3] * mx  # * self.ev.pca_dic["variance"][0]
    V0_size = ((np.array([x, y, z]) ** 2).sum(axis=0) ** 0.5).max()
    print(V0_size)
    V0tag = mat_rotation(st.dic["angle"],V0)
    ax0.plot(V0[0], V0[1], "r",zorder=0)
    props = dict(boxstyle='round', facecolor='wheat', alpha=1)
    ax0.text(V0[0,0]-V0_size*0.1, V0[1,0]-V0_size*0.1, f"AZ = N{int(round(st.dic['angle'], 0))}" + r'$^\circ$E',
        transform=ax0.transData, fontsize=10,verticalalignment='top', bbox=props)



    ax1.plot(V0tag[1], V0tag[2], "r--",zorder=0)

    ax1.text(V0tag[1, 0] - V0_size * 0.1, V0tag[2, 0] - V0_size * 0.03,
             r'$\phi=$'+ f"{90 -int(round(st.dic['inclination'], 0))}" + r'$^\circ$',
             transform=ax1.transData, fontsize=10, verticalalignment='top', bbox=props)
    ax1.set_xlim(-V0_size,V0_size)
    ax0.set_xlim(-V0_size, V0_size)
    ax1.set_ylim(-V0_size, V0_size)
    ax0.set_ylim(-V0_size, V0_size)

    ax0.annotate('N'+r'$^\circ$', xy=(0.05, 1), xycoords='axes fraction',
                xytext=(0.05, 0.7), textcoords='axes fraction',
                arrowprops=dict(facecolor='black', shrink=0.05),
                horizontalalignment='center', verticalalignment='top',
                )

    ax0.annotate('E' + r'$^\circ$', xy=(1, 0.95), xycoords='axes fraction',
                 xytext=(0.7, 0.95), textcoords='axes fraction',
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 horizontalalignment='center', verticalalignment='top',
                 )

    ax1.annotate('Depth', xy=(0.07, 0.7), xycoords='axes fraction',
                 xytext=(0.07, 0.98), textcoords='axes fraction',
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 horizontalalignment='center', verticalalignment='top',
                 )

    ax1.annotate(f"N{int(round(st.dic['angle'], 0))}" + r'$^\circ$E', xy=(1, 0.95), xycoords='axes fraction',
                 xytext=(0.7, 0.95), textcoords='axes fraction',
                 arrowprops=dict(facecolor='black', shrink=0.05),
                 horizontalalignment='center', verticalalignment='top',
                 )

    ax1.axes.get_xaxis().set_visible(False)
    ax1.axes.get_yaxis().set_visible(False)

    ax0.axes.get_xaxis().set_visible(False)
    ax0.axes.get_yaxis().set_visible(False)

    # hor_proj = np.zeros(V0.shape)
    # hor_proj[:-1, :] = V0[:-1, :].copy()
    # hor_proj = hor_proj * V0_size / (hor_proj ** 2).sum() ** 0.5
    # ax0.plot3D(hor_proj[0], hor_proj[1], hor_proj[2], "r--")


    ax0.set_aspect('equal', adjustable='box')
    ax1.set_aspect('equal', adjustable='box')



def plot_3d_motion(st, data):
    # print(data.pickTime)
    fig = plt.figure(constrained_layout=False, figsize=(12, 6))
    gs1 = fig.add_gridspec(nrows=3, ncols=3, left=0.05, right=0.48, wspace=0.05)
    ax0 = fig.add_subplot(gs1[:, :], projection='3d')

    gs2 = fig.add_gridspec(nrows=3, ncols=3, left=0.55, right=0.98, hspace=0.05)
    ax1 = fig.add_subplot(gs2[: 1, :])
    ax2 = fig.add_subplot(gs2[1: 2, :], sharex=ax1)
    ax3 = fig.add_subplot(gs2[2:, :], sharex=ax1)
    axes = [ax1, ax2, ax3]

    xn, yn, zn, deltimesn, timesn = st.data[:, st.noise_inds[0]:st.wave_inds[-1]]
    x, y, z, deltimesn_, times_ = st.data[:, st.wave_inds]

    times = st.data[
            4,
            st.wave_inds[0] - int(st.stats.sampling_rate * 0.5):st.wave_inds[-1] + int(st.stats.sampling_rate * 0.25)
    ].astype("datetime64[ms]")

    delta_times = st.data[
        3,
        st.wave_inds[0] - int(st.stats.sampling_rate * 0.5):st.wave_inds[-1] + int(st.stats.sampling_rate * 0.25)
    ]
    # print(st.stats)
    t = range(len(z))

    mx = np.sqrt((st.data[:3, st.noise_inds[0]:st.wave_inds[-1]] ** 2).sum(axis=0)).max()

    # ax.plot3D(xn, yn, zn, 'gray', zorder=0,lw=0.5)
    ax0.plot3D(x, y, z, 'gray', zorder=1, lw=0.5)
    lim_val = max(np.sqrt(x**2+y**2+z**2))*1.2
    sct = ax0.scatter3D(x, y, z, c=t, cmap=cm, zorder=2)

    pca_vec = np.c_[np.ravel(st.dic["components"])]
    # pca_vec /= np.sqrt((pca_vec**2).sum())
    pca_vec = np.concatenate((pca_vec, np.zeros(pca_vec.shape)), axis=1)
    V0 = pca_vec[:3] * mx  # * self.ev.pca_dic["variance"][0]
    V0_size = (V0**2).sum()**0.5
    ax0.plot3D(V0[0], V0[1], V0[2], "r--", zorder=0)

    hor_proj = np.zeros(V0.shape)
    hor_proj[:-1, :] = V0[:-1, :].copy()
    hor_proj = hor_proj*V0_size/(hor_proj ** 2).sum() ** 0.5
    ax0.plot3D(hor_proj[0], hor_proj[1], hor_proj[2], "r", zorder=0)

    epi = convert_ang_to_xy(data.epi_direction, 1)
    epi = epi[0]
    epi = np.c_[np.array([np.ravel(epi)]), 0]
    epi = (epi / (epi ** 2).sum() ** 0.5) * (hor_proj ** 2).sum() ** 0.5  # )
    epi = np.append(epi, np.zeros(epi.shape), axis=0).T
    # ax0.plot3D(epi[0], epi[1], epi[2], "g")
    lims = [-lim_val,lim_val]
    # ax0.set_xlim(-lim_val,lim_val)
    # ax0.set_ylim(-lim_val, lim_val)
    # ax0.set_zlim(-lim_val, lim_val)
    ax0.set_xlim3d(lims)
    ax0.set_ylim3d(lims)
    ax0.set_zlim3d(lims)

    arrows = arrow_dic(lim_val)

    ax0.arrows = []
    ax0.labels = []
    aN = mpatches.FancyArrowPatch((-lim_val, 0), (-lim_val, -lim_val))  # , val[1][1] - val[1][0])
    aE = mpatches.FancyArrowPatch((-lim_val, -lim_val), (0, -lim_val))
    aZ = mpatches.FancyArrowPatch((-lim_val, lim_val), (-lim_val, 0))
    ax0.add_patch(aN)
    art3d.pathpatch_2d_to_3d(aN, z=lim_val, zdir="z")
    ax0.add_patch(aE)
    art3d.pathpatch_2d_to_3d(aE, z=lim_val, zdir="z")

    ax0.add_patch(aZ)
    art3d.pathpatch_2d_to_3d(aZ, z=-lim_val, zdir="x")
    for key, val in arrows.items():
        # a = Arrow3D(
        #     [val[0][0], val[0][1]],
        #     [val[1][0], val[1][1]],
        #     [val[2][0], val[2][1]],
        #     mutation_scale=20,
        #     arrowstyle='-|>', color='k'
        # )
        # ax0.arrows.append(ax0.add_artist(a))

        t = ax0.text(
            val[0][1],
            val[1][1],
            val[2][1],
            key
        )
        ax0.labels.append(t)
    colnames = st.stats.columnNames[1:-1].split(",")
    for ii,[ax, trace] in enumerate(zip(axes,st.data[:3])):

        # print(st.stats.columnNames)
        # print(type(st.stats.columnNames))
        # print(st.stats.instrument[:-1]+st.stats.columnNames[ii])
        # print(times_[-1].astype("datetime64[ms]"),st.pick)

        #####
        # ax.plot(st.data[4].astype("datetime64[ms]"),trace,"k",lw=0.5)
        # ax.axvspan(st.pick, times_[-1].astype("datetime64[ms]"), alpha=0.1, color='green')
        # ax.axvline(x=st.pick,c="r")
        # ax.axvline(x=times_[-1].astype("datetime64[ms]"), c="b")
        #####
        ax.plot(st.data[3], trace, "k", lw=0.5)
        ax.axvspan(deltimesn_[0], deltimesn_[-1], alpha=0.1, color='green')
        ax.axvline(x=deltimesn_[0], c="r")
        ax.axvline(x=deltimesn_[-1], c="b")

        ax.axhline(y=st.noise_threshold,c="lime",ls="--")
        ax.axhline(y=-st.noise_threshold, c="lime",ls="--")

        ax.annotate(st.stats.instrument[:-1]+colnames[ii], xy=(0, 0), xytext=(0.05, 0.8), fontsize=10,
                         xycoords='axes fraction', textcoords='axes fraction',
                         bbox=dict(facecolor='white', alpha=0.8),
                         horizontalalignment='left', verticalalignment='top')

    axes[2].get_shared_x_axes().join(axes[0],axes[1],axes[2])
    axes[2].get_shared_y_axes().join(axes[0], axes[1], axes[2])

    axes[2].set_xlim(delta_times[0],delta_times[-1])
    axes[2].set_xlabel("Time [s]")
    fig.suptitle(st.stats.event)


    # if Type == "3d":
    #     ax0 = fig.add_subplot(gs1[:, :], projection='3d')

class P_wave_analysis:

    def __init__(self, station, trace_path, Type=None):
        self.get_sta_data(station)

        # data = np.zeros([len(self.pick_df), 6])

        self.streams = {}
        self.array_data = {
            "data": {},
            "noise": {},
            "names": {}
        }
        # self.array_data_noise = {}
        # self.array_data_names = {}
        # i = 0
        for num,file in enumerate(self.pick_df.evid.values):
            # print(num,file)
            st = trace_data(trace_path+file+".mseed", file, Type=Type)
            # print(len(st.noise_inds))
            st.pick = self.pick_df[self.pick_df.evid == file].iloc[0].pickTime
            st.pick_float = self.pick_df[self.pick_df.evid == file].iloc[0].pickTime_floats
            inx_pick = find_nearest(st.data[4], st.pick_float)
            self.compute_noise_level(st)
            st.noise_threshold = st.noise_level*self.noise_multiple
            if abs(st.noise_threshold) < abs(st.data[2,inx_pick])*2:
                st.noise_threshold = abs(st.data[2,inx_pick])*2


            inx_end = inx_pick
            tag = None
            for _ in range(self.half_cycle_num):
                inx_end,tag = wavelength(
                    inx_end,
                    st.noise_threshold,
                    st.data[2],
                    end_index=inx_end + 0.3 * st.stats.sampling_rate,
                    final=tag
                )
            # if self.half_cycle_num!=2:
            #     for _ in range(self.half_cycle_num):
            #         inx_end = wavelength_test(inx_end, st.noise_threshold, st.data[2],end_index=len(st.data[2]))
            # else:
            #     inx_end = wavelength(inx_end, st.noise_threshold, st.data[2], end_index=len(st.data[2]))[0]
            #inx_end = inx_pick + size_of_fast(st.data[2,inx_pick:],st.noise_level,self.noise_multiple,self.half_cycle_num)
            st.wave_inds = np.arange(inx_pick,inx_end)
            str_arr = st.data[:3, st.wave_inds]
            str_arr_noise = st.data[:3,st.noise_inds]
            if not str(str_arr.shape[1]) in self.array_data["data"].keys():
                self.array_data["data"][str(str_arr.shape[1])] = [str_arr]
                self.array_data["noise"][str(str_arr.shape[1])] = [str_arr_noise]
                self.array_data["names"][str(str_arr.shape[1])] = [st.stats.event]
            else:
                self.array_data["data"][str(str_arr.shape[1])].append(str_arr)
                self.array_data["noise"][str(str_arr.shape[1])].append(str_arr_noise)
                self.array_data["names"][str(str_arr.shape[1])].append(st.stats.event)

            self.streams.update({file: st})
        dats = []
        components = []
        Ss = []
        V = []
        for key in self.array_data["data"].keys():
            names = np.array(self.array_data["names"][key],dtype=int)
            temp = np.array(self.array_data["data"][key])
            temp_c = (temp.copy().T - temp.mean(axis=2).T).T
            u, s, vh = np.linalg.svd(temp_c, full_matrices=False)
            # print(s)
            norms = np.where(u[:, 2, 0] < 0, 1, -1)
            temp_normalized = u.transpose((0, 2, 1)) * norms[:, np.newaxis, np.newaxis]

            # angles = np.round((np.degrees(np.arctan2(temp_normalized[:, 0, 0], temp_normalized[:, 0, 1]))) % 360, 1)
            # inclinations = np.absolute(np.degrees(np.arcsin(temp_normalized[:, 0, -1])))
            # Ps = np.round(1 - ((s[:, 1] + s[:, 2]) / (2 * s[:, 0])), 2)
            # Pp = np.round(1 - s[:, 2] / s[:, 1], 2)
            # temp = np.array(self.array_data["data"][key])
            temp_noise = np.array(self.array_data["noise"][key])
            snr = (np.abs(temp.T) - np.abs(temp.T).mean(axis=(0, 1))).std(axis=(0, 1)) / \
                  (np.abs(temp_noise.T) - np.abs(temp_noise.T).mean(axis=(0, 1))).std(axis=(0, 1))
            # dat = np.array([names,angles,Ps,Pp,snr,inclinations]).T
            dat = np.array([names,snr]).T
            dats.append(dat)
            Ss.append(s)
            components.append(temp_normalized)
            V+=list(vh)
        # print(len(V))
        dats = np.concatenate(dats, axis=0)
        components = np.concatenate(components, axis=0)
        Ss = np.concatenate(Ss, axis=0)
        components = np.array(components)[np.argsort(dats[:, 0])]
        Ss = np.array(Ss)[np.argsort(dats[:, 0])]
        V = [V[i] for i in list(np.argsort(dats[:, 0]))]
        data = dats[np.argsort(dats[:, 0])]

        angles = np.round((np.degrees(np.arctan2(components[:, 0, 0], components[:, 0, 1]))) % 360, 1)
        inclinations = np.absolute(np.degrees(np.arcsin(components[:, 0, -1])))
        Ps = np.round(1 - ((Ss[:, 1] + Ss[:, 2]) / (2 * Ss[:, 0])), 2)
        Pp = np.round(1 - Ss[:, 2] / Ss[:, 1], 2)
        data = np.concatenate(
            (data, angles[:, np.newaxis], inclinations[:, np.newaxis], Ps[:, np.newaxis], Pp[:, np.newaxis]), axis=1
        )
        self.dic = {str(int(name[0])): comp for name,comp in zip(data,components)}

        # self.Vdic = {str(int(name[0])): comp for name,comp in zip(data,V)}


        df =  pd.DataFrame(data,columns=["evid", "snr", "angle", "inclination", "Ps", "Pp"])
        df["evid"] = df["evid"].astype(int).astype(str)
        df = self.pick_df.merge(df, on="evid")

        df["epi_direction"] = df.apply(lambda x: initial_bearing(self.lon, self.lat, x["lon"], x["lat"]),axis=1)
        df["epi_distance"] = df.apply(lambda x: haversine(self.lon, self.lat, x["lon"], x["lat"]), axis=1)
        df["offset"] = df.apply(lambda x: offset_calc(x["angle"], x["epi_direction"]),axis=1)
        self.data = df
        # print(len(self.array_data["50"]))


        # print(snr)


    @classmethod
    def set_filter(cls,min_max_HZ):
        cls.min_max_HZ = min_max_HZ
        trace_data.set_filter_limits(min_max_HZ[0],min_max_HZ[1])

    @classmethod
    def set_noise_multiple(cls, noise_multiple):
        cls.noise_multiple = noise_multiple

    @classmethod
    def set_catalog(cls, cat_path, bounds=None):
        """

        :param cat: catalog [csv]
        :param bounds: [min_lon, max_lon, min_lat, max_lat] *optional
        :return:
        """
        cat = pd.read_csv(cat_path, dtype={"evid": str})
        trace_data.set_origins(cat_path)
        if bounds:
            cls.bounds = bounds
            cat = cat[
                (cat.lon >= bounds[0]) &
                (cat.lon <= bounds[1]) &
                (cat.lat >= bounds[2]) &
                (cat.lat <= bounds[3])
            ]
        cls.cat = cat


    @classmethod
    def set_picks(cls,picks, type, area=None):
        """

        :param picks:
        :param type: "P" or "S"
        :return:
        """
        pick_df = pd.read_csv(picks, dtype={"event": str,"evid":str})
        pick_df = pick_df[pick_df.Phase == type]
        if hasattr(cls, "time_before"):
            pick_df.pickTime = pd.to_datetime(pick_df.pickTime) - cls.time_before
        times = np.array(pick_df.pickTime.tolist(), dtype="datetime64[ms]")
        pick_df["pickTime_floats"] = times.astype("float")
        if "event" in pick_df.columns:
            pick_df.rename({'event': 'evid'}, axis=1, inplace=True)
        # print(pick_df.dtypes)
        pick_df = pick_df.merge(cls.cat, left_on="evid", right_on="evid")
        pick_df.loc[pick_df.depth < 5, "depth"] = 5
        pick_df.loc[pick_df.depth > 30, "depth"] = 30
        if area:
            pick_df = pick_df[
                (pick_df.lon >= area[0]) &
                (pick_df.lon <= area[1]) &
                (pick_df.lat >= area[2]) &
                (pick_df.lat <= area[3])
            ]
        cls.pick_df = pick_df

    @classmethod
    def set_sites(cls, path):
        df = pd.read_csv(path)
        df.rename({"longitude": "lon", "latitude": "lat"}, axis=1, inplace=True)
        cls.site_df = df

    @classmethod
    def set_half_cycle_num(cls, num):
        cls.half_cycle_num = num

    @classmethod
    def set_time_before_pick(cls, time):
        cls.time_before = np.timedelta64(time,"ms")

    @classmethod
    def set_time_for_noise_level(cls, time):
        cls.time_for_noise_level = time#np.timedelta64(time,"s")

    def get_sta_data(self, sta):
        sta = self.site_df[self.site_df.station == sta].iloc[0]
        self.__dict__.update(sta)


    def compute_noise_level(self, tr):
        tw = np.array([-self.time_for_noise_level, 0], dtype="timedelta64[s]")
        tw = tr.pick + tw
        tw = tw.astype(float) * 1e-6
        inds = np.ravel(np.argwhere(
            (tr.data[4, :] >= tw[0]) &
            (tr.data[4, :] < tw[1])
        ))
        tr.noise_level = (np.abs(tr.data[2,inds]) - np.abs(tr.data[2,inds]).mean()).std()
        tr.noise_inds = inds

