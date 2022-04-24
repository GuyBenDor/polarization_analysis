from matplotlib import pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
import os
import pandas as pd
from functions.baseFunctions import sliding_window, find_nearest, arrow_dic, traces
import scipy
from scipy.signal import find_peaks
import matplotlib.patches as mpatches
import mpl_toolkits.mplot3d.art3d as art3d
import time


def size_of_fast(a, noise, noise_multiple, wavelength=1):
    a_noise = noise * noise_multiple
    peaks_s, properties = find_peaks(a, prominence=2*a_noise)
    peaks_s1, properties = find_peaks(-a, prominence=2*a_noise)
    peaks_s = np.sort(np.concatenate([peaks_s,peaks_s1]))
    a_check = a if a[peaks_s[wavelength]] > 0 else -a
    cond = True
    inx = peaks_s[wavelength]

    while cond:
        inx += 1
        if a_check[inx]<0:
            cond = False
        elif inx == len(a_check)-1:
            cond = False
    return inx+1

def get_init_dir(arr):
    U, s, V = np.linalg.svd(arr)
    dic = {}
    if s.shape[1] == 3:
        Ps = 1 - ((s[:, 1]+s[:, 2])/(2*s[:, 0]))
        inclination = np.degrees(np.arcsin(U[:, 2, 0]))
        dic.update({
            "Ps": Ps,
            "Inclination": inclination
        })
    angle = np.round(np.degrees(np.arctan2(U[:,0,0],U[:,1,0]))%360,1)
    Pp = 1 - s[:, -1] / s[:, -2]
    dic.update({"angle":angle,"Pp":Pp,"U":U})
    return dic



def slide_correlation(a,b,first_high_coef=None, remove_av=False):
    b_st = sliding_window(b, len(a))
    a_st = np.tile(a,(b_st.shape[0],1))
    # print(b)
    # print(b_st.shape)
    # print(a_st.shape)
    con = np.dstack((a_st,b_st))
    # print(con.shape)
    con = np.einsum('ijk->ikj', con)
    # print(con.shape)

    av = con.mean(axis=2)
    if remove_av:
        correfs = np.array([np.corrcoef((con[i].T-av[i]).T)[0,1] for i in range(len(con))])
    else:
        correfs = np.array([np.corrcoef(con[i])[0, 1] for i in range(len(con))])

    if not first_high_coef or not np.any(np.abs(correfs)>=first_high_coef):
        shift_ind = np.abs(correfs).argmax()
        # print(shift_ind)
    else:
        shift_ind = np.argwhere(np.abs(correfs)>=first_high_coef)[0,0]

    # shift_ind = np.abs(correfs).argmax()
    # try:
    #     shift_ind = np.argwhere(np.abs(correfs) >= 0.89)[0][0]
    # except:
    #     shift_ind = np.abs(correfs).argmax()
    return shift_ind,correfs#[shift_ind]

# def index_for_init_motion(init_motion_smp,pickTime,timeSeries):
#     # init_motion_smp = int(self.init_motion_time * obj.stats.sampling_rate)
#     pick_i = find_nearest(np.datetime64(pickTime), timeSeries.astype("datetime64[ms]"))
#     # print(pick_i)
#     return np.array([0, init_motion_smp],dtype=int) + pick_i



# class traces(trace_data):
#     def __init__(self,path, event):
#         print(self.pick_df)
#         super().__init__(path, event)
#
#     @classmethod
#     def set_picks(cls,path):
#         df = pd.read_csv(path, dtype={"event": str})
#         df["pickTime"] = df["pickTime"].str.replace(' ', 'T')
#         df["pickTime"] = pd.to_datetime(df.pickTime)
#         cls.pick_df = df



class event_data:
    def __init__(self, dic):
        self.hypo_dist = None
        self.__dict__.update(dic)

class corr_rotated:

    def __init__(self, station_dic, init_motion_time=0.05, time_template=0.3, remove_av=False, plot_fig=False,
                 Type='2d',plot_inside=True, **kwargs):

        self.remove_av = remove_av
        self.Type = Type
        self.init_motion_time = init_motion_time
        self.time_template = time_template
        self.station_dic = station_dic
        self.plot_inside = plot_inside
        pick_df = traces.pick_df[traces.pick_df.station.isin(list(self.station_dic.keys()))].copy()
        pick_df.drop(["plt_num"], axis=1, inplace=True)
        pick_df = pick_df[pick_df.Phase == "S"]

        if 'specify_events' in kwargs:
            pick_df = pick_df[pick_df.evid.isin(kwargs['specify_events'])]
        pick_df.reset_index(inplace=True, drop=True)
        trace_parm = pd.DataFrame(self.station_dic).T
        pick_df = pick_df.merge(trace_parm, right_index=True, left_on="station")
        # print(pick_df)
        pick_df["path"] = pick_df[["station", "instrument", "evid"]].apply(
            lambda x: os.path.join(self.trace_path, x[0], x[1], f'{x[2]}.mseed'), axis=1
        )
        self.pick_df = pick_df

        self.pick_df["stream"] = pick_df[["path","evid","fmin","fmax"]].apply(
            lambda x: traces(x[0], evid=x[1], fmin=x[2],fmax=x[3]), axis=1
        )
        self.pick_df.drop(['Phase','fmax', 'fmin', 'instrument','path'], axis=1, inplace=True)
        # self.pick_df["data"] = pick_df["stream"].apply(lambda x: x.data)
        # self.pick_df["stream_length"] = pick_df["data"].apply(lambda x: x.shape[1])

        self.pick_df["pickIndex"] = self.pick_df[["pickTime","stream"]].apply(
            lambda x: find_nearest(np.datetime64(x[0]),x[1].data[4].astype("datetime64[ms]")),axis=1
        )

        self.set_arrays()
        self.pick_df.drop(['rotation_matrix', 'pickIndex'], axis=1, inplace=True)
        if plot_fig:
            for i in range(len(self.pick_df)):
                row = self.pick_df.iloc[i]
                print(row[['evid', 'Inclination_3d','angle_2d', 'angle_3d', 'Pp_2d', 'Pp_3d', 'Ps_3d','time_delay','corr_coef']])
                self.generate_delay_plot(
                    row['stream'], row['rotated_data'], row['shift_data'], row['tw'],
                    row['pca_dat'], row['U_3d'], row['U_2d'], row['time_delay'], row[f'angle_{Type}']
                    # val['2d']['U'][i], val['time_delay'][i], val[Type]['angle'][i]
                )

    def set_arrays(self):
        self.pick_df["pca_dat"] = self.pick_df[["stream", "pickIndex"]].apply(
            lambda x: x[0].data[:, x[1]:x[1]+int(self.init_motion_time * x[0].stats.sampling_rate)], axis=1
        )#np.array([0, init_motion_smp], dtype=int) + pick_i
        arr = np.array(self.pick_df.pca_dat.tolist())
        arr -= np.einsum("ijk->ikj", arr.mean(axis=2)[:, np.newaxis])  # removing mean

        for num in [2,3]:
            temp = get_init_dir(arr[:, :num])
            for key,val in temp.items():
                if key=="U":
                    val = list(val)
                self.pick_df[f'{key}_{num}d'] = val
        angle = np.array(self.pick_df[f'angle_{self.Type}'].tolist())
        # arr = np.array(self.pick_df.stream.apply(lambda x: x.data.copy()).tolist())

        c, s = np.cos(np.radians(angle)), np.sin(np.radians(angle))
        R = np.zeros([len(angle), 2, 2])
        R[:, 0, 0] = c
        R[:, 0, 1] = -s
        R[:, 1, 0] = s
        R[:, 1, 1] = c
        self.pick_df["rotation_matrix"] = list(R)

        # start = time.time()
        # grouped = self.pick_df.copy().groupby("stream_length")
        # check = []
        # for name,group in grouped:
        #     data = np.array(group.data.tolist())
        #     data[:, :2] = np.matmul(R[group.index.values], data[:, :2])
        #     group['rotated_data'] = list(data)
        #     check.append(group)
        # check = pd.concat(check, axis=0)[["evid", "rotated_data"]]
        # print(check)
        # end = time.time()
        # print("1",end - start)

        start = time.time()
        self.pick_df['rotated_data'] = self.pick_df[["rotation_matrix","stream"]].apply(
            lambda x: np.vstack([np.array(x[0])@x[1].data.copy()[:2],x[1].data.copy()[2:]]),axis=1
        )
        end = time.time()
        # print(self.pick_df)
        # print("2",end-start)

        # arr[:, :2, :] = np.matmul(R, arr[:, :2, :])
        # self.pick_df['rotated_data'] = list(arr)
        self.pick_df[['time_delay', 'shift_sample', 'ts', 'corr_coef','tw','shift_data']] = self.pick_df[[
            'rotated_data', 'stream', 'pickTime', 'pickIndex'
        ]].apply(
            lambda x: self.smallTimeWindow(x[0], x[1].data, x[2], x[3], x[1].stats.delta), axis=1, result_type='expand'
        )

    def smallTimeWindow(self, rotated, data, pickTime, pickIndex, delta):
        slow, fast, times = rotated[0],rotated[1], data[4].astype("datetime64[ms]")

        tw = np.array([
            -int(1000),  # noise start
            0,  # pick start
            0,  # end fast (small)
            int(self.time_template * 1000),  # end slow (big)

        ], dtype="timedelta64[ms]")
        tw += pickTime
        # parms = np.zeros((len(slow),4))
        # shift_data = []
        inx_noise = np.where(
            (times >= tw[0]) &
            (times <= tw[1])
        )[0]
        noise_in_fast = (np.abs(fast[inx_noise]) - np.abs(fast[inx_noise]).mean()).std()
        inx_end = size_of_fast(fast[pickIndex:], noise_in_fast, 3, self.half_cycle_num - 1)
        ts = inx_end * delta
        if hasattr(self, "minMax"):
            if self.minMax[0] > ts:
                ts = self.minMax[0]
            elif ts > self.minMax[1]:
                ts = self.minMax[1]
        tw[2:] += ts

        inx_big = np.where(
            (times >= tw[1]) &
            (times <= tw[3])
        )[0]

        inx_small = np.where(
            (times >= tw[1]) &
            (times <= tw[2])
        )[0]
        shift_sample, coef = slide_correlation(fast[inx_small], slow[inx_big], remove_av=self.remove_av)
        time_delay = (int(shift_sample) * delta).astype("float")*1e-3#total_seconds()
        start_of_slow = inx_big[0] + shift_sample
        end_of_slow = start_of_slow + len(inx_small)
        shift_data = np.zeros([2, len(inx_small)])
        shift_data[:] = [slow[start_of_slow:end_of_slow], fast[inx_small]]
        corr_coef = round(coef[shift_sample], 2)
        return [time_delay, shift_sample, ts, corr_coef,tw,shift_data]

    def generate_delay_plot(self, obj, rotated_data, shift_data, tw, smps_of_pca, U3d, U2d, time_delay, angle):
        # print("11111111111111")
        slow, fast = rotated_data[:2]

        fast_inx = np.where(
            (obj.data[4].astype("datetime64[ms]") >= tw[1]) &
            (obj.data[4].astype("datetime64[ms]") <= tw[2])
        )[0]

        slow_inx = np.where(
            (obj.data[4].astype("datetime64[ms]") >= tw[1]) &
            (obj.data[4].astype("datetime64[ms]") <= tw[3])
        )[0]
        # print(fast[fast_inx])
        # print(slow[slow_inx])

        fig = plt.figure(figsize=[18, 7])
        plt.suptitle(f'{obj.stats.station} - {obj.stats.event}')

        gs = GridSpec(12, 12, figure=fig)
        axes = [fig.add_subplot(gs[:, :4], projection='3d')]
        axes += [fig.add_subplot(gs[:6, 4:8])]
        axes += [fig.add_subplot(gs[6:8, 4:8])]
        axes += [fig.add_subplot(gs[8:10, 4:8])]
        axes += [fig.add_subplot(gs[10:12, 4:8])]
        axes += [fig.add_subplot(gs[:, 8:])]

        axes[0].plot(
            obj.data[0, fast_inx[0] - 150:fast_inx[-1]],
            obj.data[1, fast_inx[0] - 150:fast_inx[-1]],
            obj.data[2, fast_inx[0] - 150:fast_inx[-1]], "k",
            lw=0.5
        )

        axes[0].set_xlabel("E-W")
        axes[0].set_ylabel("N-S")
        # axes[0].grid(False)
        axes[0].scatter(smps_of_pca[0],smps_of_pca[1],smps_of_pca[2],
                        c="r", lw=1, label=f"first {self.init_motion_time*obj.stats.sampling_rate} samples", alpha=0.6
                        )

        r = np.sqrt((smps_of_pca ** 2).sum(axis=0)).max()
        lim = np.sqrt((obj.data[:2, fast_inx] ** 2).sum(axis=0)).max() * 1.1

        vec = np.zeros([2, 3])
        vec[1] = U3d[:, 0] * lim
        axes[0].plot(vec.T[0], vec.T[1], vec.T[2], "r")

        vec = np.zeros([2, 3])
        vec[0, :2] = -U2d[:, 0] * lim
        vec[1, :2] = U2d[:, 0] * lim
        axes[0].plot(vec.T[0], vec.T[1], vec.T[2], lw=3, zorder=0, alpha=0.3, label="inital direction")
        axes[0].set_xlim(-lim, lim)
        axes[0].set_ylim(-lim, lim)
        axes[0].set_zlim(-lim, lim)
        aN = mpatches.FancyArrowPatch((-lim, 0), (-lim, -lim))  # , val[1][1] - val[1][0])
        aE = mpatches.FancyArrowPatch((-lim, -lim), (0, -lim))
        aZ = mpatches.FancyArrowPatch((-lim, lim), (-lim, 0))
        axes[0].add_patch(aN)
        art3d.pathpatch_2d_to_3d(aN, z=lim, zdir="z")
        axes[0].add_patch(aE)
        art3d.pathpatch_2d_to_3d(aE, z=lim, zdir="z")
        axes[0].add_patch(aZ)
        art3d.pathpatch_2d_to_3d(aZ, z=-lim, zdir="x")
        arrows = arrow_dic(lim)
        for key, val in arrows.items():
            t = axes[0].text(
                val[0][1],
                val[1][1],
                val[2][1],
                key
            )
            # axes[0].labels.append(t)
        axes[1].plot(obj.data[3] + time_delay, fast, "k", lw=0.5)
        axes[1].plot(obj.data[3], slow, "k", lw=0.5)

        fast_string = f"fast N{int(round(angle, 0))}" + r'$^\circ$E'
        slow_string = f"slow N{int(90 + round(angle, 0)) % 360}" + r'$^\circ$E'
        delay_string = f"{int(time_delay * 1000)}ms delay time applied"

        axes[1].plot(obj.data[3, fast_inx] + time_delay, fast[fast_inx],
                     label=fast_string, color='red')
        axes[1].plot(obj.data[3, slow_inx], slow[slow_inx], label=slow_string, color='green')

        axes[1].axvspan(obj.data[3, slow_inx[0]], obj.data[3, slow_inx[-1]], alpha=0.1, color='green')
        axes[1].axvspan(obj.data[3, fast_inx[0]] + time_delay, obj.data[3, fast_inx[-1]] + time_delay,
                        alpha=0.1, color='red')

        axes[1].annotate(delay_string, xy=(0, 0), xytext=(0.05, 0.1), fontsize=10,
                         xycoords='axes fraction', textcoords='axes fraction',
                         bbox=dict(facecolor='white', alpha=0.8),
                         horizontalalignment='left', verticalalignment='top')
        axes[1].legend(loc="upper right")

        axes[2].plot(obj.data[3], fast, "k", lw=0.5)
        axes[2].axvspan(obj.data[3, fast_inx[0]], obj.data[3, fast_inx[-1]], alpha=0.1, color='red')
        # axes[2].axhline(y= self.noise_in_fast,c="grey")
        # axes[2].axhline(y=-self.noise_in_fast, c="grey")
        axes[2].annotate(fast_string, xy=(0, 0), xytext=(0.05, 0.3), fontsize=10,
                         xycoords='axes fraction', textcoords='axes fraction',
                         bbox=dict(facecolor='white', alpha=0.8),
                         horizontalalignment='left', verticalalignment='top')

        axes[3].plot(obj.data[3], slow, "k", lw=0.5)
        axes[3].axvspan(obj.data[3, slow_inx[0]], obj.data[3, slow_inx[-1]], alpha=0.1, color='green')
        axes[3].annotate(slow_string, xy=(0, 0), xytext=(0.05, 0.3), fontsize=10,
                         xycoords='axes fraction', textcoords='axes fraction',
                         bbox=dict(facecolor='white', alpha=0.8),
                         horizontalalignment='left', verticalalignment='top')
        # self.axes[i].axvspan(self.data[3, inx_s], self.data[3, inx_e], alpha=0.1, color='red')
        axes[4].plot(obj.data[3], obj.data[2], "k", lw=0.5)
        axes[4].annotate("vertical", xy=(0, 0), xytext=(0.05, 0.3), fontsize=10,
                         xycoords='axes fraction', textcoords='axes fraction',
                         bbox=dict(facecolor='white', alpha=0.8),
                         horizontalalignment='left', verticalalignment='top')

        y_lim = np.abs(np.concatenate([fast[fast_inx], slow[slow_inx]])).max() * 1.2
        for i in range(1, 5):
            axes[i].set_xlim(obj.data[3, slow_inx[0] - 20], obj.data[3, slow_inx[-1] + 30])
            axes[i].set_ylim(-y_lim, y_lim)

        axes[4].get_shared_x_axes().join(axes[1], axes[2], axes[3], axes[4])
        axes[4].get_shared_y_axes().join(axes[1], axes[2], axes[3], axes[4])
        axes[4].set_xlabel("Time [s]")

        lim = np.sqrt((shift_data[:] ** 2).sum(axis=0)).max() * 1.1
        axes[5].plot(shift_data[0], shift_data[1], "k",
                     lw=0.5)  # ,self.data[2,start_of_slow:end_of_slow])
        axes[5].set_xlim(-lim, lim)
        axes[5].set_ylim(-lim, lim)
        axes[5].set_aspect('equal', adjustable='box')
        axes[5].set_title("After applying a delay to fast component")
        if self.plot_inside:
            plt.show()

    @classmethod
    def set_paths(cls,**kwargs):
        if not "trace_path" in kwargs.keys():
            print("add trace_path")
        if not "pick_path" in kwargs.keys():
            print("add pick_path")
        cls.trace_path = kwargs["trace_path"]
        traces.set_picks(os.path.join(kwargs["pick_path"]))
        # cls.pick_path = kwargs["pick_path"]

    @classmethod
    def set_sites(cls, path):
        df = pd.read_csv(path)
        df.rename({"longitude":"lon", "latitude":"lat"}, axis=1, inplace=True)
        cls.site_df = df

    @classmethod
    def set_half_cycle_num(cls,num):
        cls.half_cycle_num = num

    @classmethod
    def set_origins(cls,orig_path):
        traces.set_origins(orig_path)
        cls.orig_df = traces.orig_df

    @classmethod
    def set_small_window_limits(cls,minMax):
        cls.minMax = minMax





# class corr_rotated(trace_data):
#
#     def __init__(self,path, event, init_motion_time = 0.05, small_window_limits=[],time_template=None,
#                  fmin=None, fmax=None, remove_av = False, const_small_window_length =None,
#                  use_3d_rotation=False, **kwargs):
#         """
#
#         :param path:
#         :param event:
#         :param init_motion_smp:
#         :param time_small_window:
#         :param time_template:
#         :param pick_file_path:
#         :param fmin:
#         :param fmax:
#         :param template_starts_from_pick:
#         :param cc_search_in_forward_smp:
#         :param kwargs:
#             contains:
#                 1. small_time_window_limit: float
#                 2. determine_tw_from_hypo_dist: bool
#
#         """
#         super().__init__(path, event, fmin, fmax)
#         self.__dict__.update(kwargs)
#         self.remove_av = remove_av
#
#         # if time_template:
#         #     if time_template < time_small_window:
#         #         raise ValueError(
#         #             f"time_tamplate [{time_template}] most by bigger than time_small_window [{time_small_window}]")
#         # else:
#         #     time_template = 2 * time_small_window
#
#         self.get_picks()
#
#         self.init_motion_smp = int(init_motion_time*self.stats.sampling_rate)
#         self.const_small_window_length = const_small_window_length
#         if not self.const_small_window_length:
#             if small_window_limits:
#                 self.time_small_window_limits = np.array(small_window_limits)*1000
#                 self.time_small_window_limits = self.time_small_window_limits.astype('timedelta64[ms]')
#
#         if "determine_tw_from_hypo_dist" in kwargs and kwargs["determine_tw_from_hypo_dist"]:
#             self.time_template = self.event_data.hypo_dist * ((1 / self.v_low) - (1 / self.v_high))
#             if self.time_template < kwargs['large_time_window_limits'][0]:
#                 print(f"orig dt was: {self.time_template}")
#                 self.time_template = kwargs['large_time_window_limits'][0]
#             elif self.time_template > kwargs['large_time_window_limits'][1]:
#                 print(f"orig dt was: {self.time_template}")
#                 self.time_template = kwargs['large_time_window_limits'][1]
#             print(f"dt is evaluated at {self.time_template}")
#             # print("dt:", self.event_data.hypo_dist * (1 / 4. - 1 / 4.5))
#         else:
#             self.time_template = time_template
#
#         self.pick = self.picks["S"]["picks"].astype("datetime64[ms]")
#         # print(self.pick)
#         self.first_motion_rotation_2d()
#         ########
#         self.first_motion_rotation_3d()
#         #######
#         if use_3d_rotation:
#             self.rotated = mat_rotation(self.pca_dic_3d["rotation"], self.data[:2])
#             self.rotation_angle, self.rect_inital = self.pca_dic_3d["rotation"], self.pca_dic_3d["Ps"]
#         else:
#             self.rotated = mat_rotation(self.pca_dic_2d["rotation"], self.data[:2])
#             self.rotation_angle, self.rect_inital = self.pca_dic_2d["rotation"], self.pca_dic_2d["Ps"]
#
#         # self.cross_correlation_func(template_starts_from_pick,cc_search_in_forward_smp)
#         self.cross_corr_func_new(**kwargs)
#
#     def first_motion_rotation_2d(self):
#         # pick = self.picks["S"]["picks"].astype("datetime64[ms]")
#         pick_i = find_nearest(self.pick, self.data[4].astype("datetime64[ms]"))
#         tw = np.array([0, self.init_motion_smp]) + pick_i
#
#         temp = self.data[:2, tw[0]:tw[1]].T
#         temp = temp - temp.mean(axis=0)  # removing mean
#
#         # pca = PCA(n_components=2).fit(temp)
#         # U,s = pca.components_, pca.explained_variance_
#
#         U, s, V = np.linalg.svd(temp.T)
#         ps1 = 1 - s[1] / s[0]
#         angs1 = np.degrees(np.arctan2(U[:, 0], U[:, 1])) % 360
#         # rot_ang =
#         # print(rot_ang)
#         # print("rotation",rot_ang, ps1)
#         # self.rotation_angle, self.rect_inital = rot_ang, ps1
#         self.pca_dic_2d = {"rotation": angs1[0], "Ps": ps1}
#         self.samples_of_pca = tw
#
#     def first_motion_rotation_3d(self):
#         # pick = self.picks["S"]["picks"].astype("datetime64[ms]")
#         pick_i = find_nearest(self.pick, self.data[4].astype("datetime64[ms]"))
#         tw = np.array([0, self.init_motion_smp]) + pick_i
#
#         temp = self.data[:3, tw[0]:tw[1]].T
#         temp = temp - temp.mean(axis=0)  # removing mean
#
#         # pca = PCA(n_components=2).fit(temp)
#         # U,s = pca.components_, pca.explained_variance_
#
#         U, s, V = np.linalg.svd(temp.T)
#         angles = np.round((np.degrees(np.arctan2(U.T[:, 0], U.T[:, 1]))) % 360, 1)
#
#         self.U = U
#         self.rotated_traces = V
#
#         inclination = np.degrees(np.arctan2(U[2,0],np.sqrt((U[:-1,0]**2).sum())))
#         Ps = round(1 - ((s[1] + s[2]) / (2*s[0])), 2)
#         Pp = round(1 - s[2] / s[1], 2)
#         self.pca_dic_3d = {"rotation": angles[0], "inclination":inclination, "Ps":Ps, "Pp":Pp}
#         print(self.pca_dic_3d)
#
#     def cross_corr_func_new(self, **kwargs):
#         """
#
#         :param kwargs:
#             1. determine_tw_from_hypo_dist: bool
#         :return:
#         """
#         [slow, fast], times = self.rotated,self.data[4].astype("datetime64[ms]")
#
#         tw = np.array([
#             -int(1000), #noise start
#             0,  # pick start
#             0,  # end fast (small)
#             int(self.time_template * 1000), # end slow (big)
#
#         ], dtype="timedelta64[ms]")
#
#         tw = self.pick + tw
#         pick_ind = find_nearest(times,tw[1])
#
#         inx_noise = np.where(
#             (times >= tw[0]) &
#             (times <= tw[1])
#         )[0]
#
#         self.noise_in_fast = (np.abs(fast[inx_noise]) - np.abs(fast[inx_noise]).mean()).std()
#
#         # print(self.noise_in_fast)
#         if self.const_small_window_length:
#             ts = np.timedelta64(self.const_small_window_length,"ms")
#         else:
#             inx_end = size_of_fast(fast[pick_ind:], self.noise_in_fast, 3,self.half_cycle_num-1)
#             #NOISE MULTIPLE, num_of_half_wavelength
#
#             # inx_end = pick_ind
#             # end_cond = False
#             # for _ in range(self.half_cycle_num):
#             #     if _ + 1 == self.half_cycle_num:
#             #         end_cond = True
#             #     inx_end = wavelength(inx_end, self.noise_in_fast*5, fast,end_condition=end_cond)
#             # inx_end -= pick_ind
#             # print(self.stats.delta)
#
#             ts = inx_end * self.stats.delta
#
#         if hasattr(self,"time_small_window_limits"):
#             if self.time_small_window_limits[0] > ts:
#                 ts = self.time_small_window_limits[0]
#             elif ts > self.time_small_window_limits[1]:
#                 ts = self.time_small_window_limits[1]
#         # hor = np.sqrt((self.rotated[:,pick_ind:pick_ind+int(ts/self.stats.delta)]**2).sum(0))
#         hor = self.rotated[:,pick_ind:pick_ind+int(ts/self.stats.delta)]
#         hor = np.sqrt(((hor - hor.mean(axis=1, keepdims=True))**2).sum(0))
#         # print(hor)
#         ver = np.abs(self.data[2,pick_ind:pick_ind+int(ts/self.stats.delta)]-self.data[2,pick_ind:pick_ind+int(ts/self.stats.delta)].mean())
#         self.hor_to_ver_ratio = hor.sum()/ver.sum()
#         print(self.hor_to_ver_ratio)
#         # print(test)
#
#         # print(ts/self.stats.delta)
#
#         # if inx_end / self.stats.sampling_rate < self.time_small_window:
#         #     inx_end = int(self.time_small_window*self.stats.sampling_rate)
#         # if hasattr(self,"small_time_window_limit"):
#         #     if inx_end / self.stats.sampling_rate > self.small_time_window_limit:
#         #         inx_end = int(self.small_time_window_limit * self.stats.sampling_rate)
#
#         # if "determine_tw_from_hypo_dist" in kwargs and kwargs["determine_tw_from_hypo_dist"]:
#         #     tw[2:] = tw[2:] + inx_end*self.stats.delta ############### <<<<<<<<<<<<<<<
#         # else:
#         #     tw[2] = tw[2] + inx_end * self.stats.delta
#         # print(inx_end * self.stats.delta)
#         # tw[2:] += inx_end * self.stats.delta
#
#         if "add_small_tw_to_time_template" in kwargs and kwargs["add_small_tw_to_time_template"]:
#             tw[2:] += ts
#         else:
#             tw[2] += ts
#
#
#         if tw[2]>tw[3]:
#             print(tw[2], tw[3])
#             raise ValueError("your small window is larger than your big window")
#         inx_big = np.where(
#             (times >= tw[1]) &
#             (times <= tw[3])
#         )[0]
#
#         inx_small = np.where(
#             (times >= tw[1]) &
#             (times <= tw[2])
#         )[0]
#         # plt.plot(fast)
#         # # plt.plot(peaks_s[peaks_s<inx], a[peaks_s[peaks_s<inx]], "x")
#         # plt.plot(inx_small, fast[inx_small], "x")
#         # plt.show()
#
#         if hasattr(self,"first_high_coef_val"):
#             shift_sample, coef = slide_correlation(
#                 fast[inx_small], slow[inx_big],
#                 first_high_coef=self.first_high_coef_val,
#                 remove_av=self.remove_av
#             )
#         else:
#             shift_sample, coef = slide_correlation(fast[inx_small], slow[inx_big], remove_av=self.remove_av)
#         # print(coef)
#         time_delay = int(shift_sample) * self.stats.delta.astype("float") * 1e-3
#         # print(coef)
#         self.time_delay, self.shift_sample, self.tw, self.fast_window_length = time_delay, shift_sample, tw, ts
#
#         start_of_slow = inx_big[0] + self.shift_sample
#         end_of_slow = start_of_slow + len(inx_small)
#         temp = np.zeros([2, len(inx_small)])
#         temp[:] = [slow[start_of_slow:end_of_slow], fast[inx_small]]
#         self.shifted_data = temp
#         back = np.zeros([2,int(self.stats.sampling_rate*1.)])
#         back[0] = slow[start_of_slow - int(self.stats.sampling_rate*1.):start_of_slow]
#         back[1] = fast[inx_small[0] - int(self.stats.sampling_rate * 1.):inx_small[0]]
#         for_rms = (temp - temp.mean(axis=0)).std()
#         back_rms = (back - back.mean(axis=0)).std()
#         # fig,ax = plt.subplots()
#         # ax.scatter(temp[0],temp[1])
#         # ax.scatter(back[0], back[1],alpha=0.5)
#         # plt.show()
#         self.snr = for_rms/back_rms
#         U, s, V = np.linalg.svd(self.shifted_data)
#         self.rect_final = 1 - s[1] / s[0]
#         self.corr_coef = round(coef[shift_sample],2)#np.round_(np.corrcoef(self.shifted_data), 2)[0, 1]
#
#     def get_picks(self):
#         df = self.pick_df[self.pick_df.event == self.stats.event]
#         if len(df) == 0:
#             print(f"\tevent {self.stats.event} has no pick")
#             return
#         sta = self.site_df[self.site_df.station==self.stats.station].iloc[0]
#         self.event_data = event_data(df.iloc[0][['orig_time', 'lat', 'lon', 'depth', 'mb', 'ms', 'ml']])
#         self.event_data.hypo_dist = np.sqrt(haversine(sta.lon,sta.lat,self.event_data.lon,self.event_data.lat)**2+
#                                             (self.event_data.depth-sta.elevation)**2)
#         # print("hypo_dist:",self.event_data.hypo_dist,"km")
#         P_picks = df[df.Phase == "P"][["pickTime_floats"]].to_numpy().T
#         S_picks = df[df.Phase == "S"][["pickTime_floats"]].to_numpy().T
#         # print(P_picks)
#         dic = {
#             "P": {"picks": P_picks[0]},
#             "S": {"picks": S_picks[0]}
#         }
#         self.picks.update(dic)
#
#     def generate_delay_plot(self,num, fig_3d=False):
#
#         slow, fast = self.rotated
#
#         fast_inx = np.where(
#             (self.data[4].astype("datetime64[ms]") >= self.tw[1]) &
#             (self.data[4].astype("datetime64[ms]") <= self.tw[2])
#         )[0]
#
#         slow_inx = np.where(
#             (self.data[4].astype("datetime64[ms]") >= self.tw[1]) &
#             (self.data[4].astype("datetime64[ms]") <= self.tw[3])
#         )[0]
#
#
#         # start_of_slow = slow_inx[0] + self.shift_sample
#         # end_of_slow = start_of_slow + len(fast_inx)
#         # temp = np.zeros([2,len(fast_inx)])
#         # temp[:] = [slow[start_of_slow:end_of_slow],fast[fast_inx]]
#
#         # axes[0].plot(self.data[0, fast_inx], self.data[1, fast_inx], 'k', lw=0.5)
#
#         fig = plt.figure(num, figsize=[18, 7])
#         plt.suptitle(self.stats.event)
#
#         gs = GridSpec(12, 12, figure=fig)
#         if fig_3d:
#             axes = [fig.add_subplot(gs[:, :4], projection='3d')]
#         else:
#             axes = [fig.add_subplot(gs[:, :4])]  # , projection='3d')]
#         axes += [fig.add_subplot(gs[:6, 4:8])]
#         axes += [fig.add_subplot(gs[6:8, 4:8])]
#         axes += [fig.add_subplot(gs[8:10, 4:8])]
#         axes += [fig.add_subplot(gs[10:12, 4:8])]
#         axes += [fig.add_subplot(gs[:, 8:])]
#
#         if fig_3d:
#             axes[0].plot(
#                 self.data[0, fast_inx[0] - 150:fast_inx[-1]],
#                 self.data[1, fast_inx[0] - 150:fast_inx[-1]],
#                 self.data[2, fast_inx[0] - 150:fast_inx[-1]], "k",
#                 lw=0.5
#             )
#
#             axes[0].set_xlabel("E-W")
#             axes[0].set_ylabel("N-S")
#             # axes[0].grid(False)
#             axes[0].scatter(self.data[0, self.samples_of_pca[0]:self.samples_of_pca[1]],
#                             self.data[1, self.samples_of_pca[0]:self.samples_of_pca[1]],
#                             self.data[2, self.samples_of_pca[0]:self.samples_of_pca[1]],
#                             c="r", lw=1, label=f"first {self.init_motion_smp} samples", alpha=0.6
#                             )
#
#             # print(vec)
#             r = np.sqrt((self.data[:3, self.samples_of_pca[0]:self.samples_of_pca[1]] ** 2).sum(axis=0)).max()
#             lim = np.sqrt((self.data[:2, fast_inx] ** 2).sum(axis=0)).max() * 1.1
#
#             vec = np.zeros([2, 3])
#             vec[1] = self.U[:,0]*lim
#             axes[0].plot(vec.T[0], vec.T[1], vec.T[2], "r")
#
#             x = np.cos(np.radians([self.rotation_angle, self.rotation_angle + 180])) * r
#             y = np.sin(np.radians([self.rotation_angle, self.rotation_angle + 180])) * r
#             axes[0].plot(y, x,np.zeros(len(x)), lw=3, zorder=0, alpha=0.3, label="inital direction")
#             axes[0].set_xlim(-lim, lim)
#             axes[0].set_ylim(-lim, lim)
#             axes[0].set_zlim(-lim, lim)
#             arrows = arrow_dic(lim)
#
#             axes[0].arrows = []
#             axes[0].labels = []
#             aN = mpatches.FancyArrowPatch((-lim, 0), (-lim, -lim))  # , val[1][1] - val[1][0])
#             aE = mpatches.FancyArrowPatch((-lim, -lim), (0, -lim))
#             aZ = mpatches.FancyArrowPatch((-lim, lim), (-lim, 0))
#             axes[0].add_patch(aN)
#             art3d.pathpatch_2d_to_3d(aN, z=lim, zdir="z")
#             axes[0].add_patch(aE)
#             art3d.pathpatch_2d_to_3d(aE, z=lim, zdir="z")
#             axes[0].add_patch(aZ)
#             art3d.pathpatch_2d_to_3d(aZ, z=-lim, zdir="x")
#             for key, val in arrows.items():
#                 # a = Arrow3D(
#                 #     [val[0][0], val[0][1]],
#                 #     [val[1][0], val[1][1]],
#                 #     [val[2][0], val[2][1]],
#                 #     mutation_scale=20,
#                 #     arrowstyle='-|>', color='k'
#                 # )
#                 # axes[0].arrows.append(axes[0].add_artist(a))
#
#                 t = axes[0].text(
#                     val[0][1],
#                     val[1][1],
#                     val[2][1],
#                     key
#                 )
#                 axes[0].labels.append(t)
#             # axes[0].set_aspect('equal', adjustable='box')
#             # axes[0].legend(loc="upper right")
#             # axes[0].annotate('North', xy=(0.1, 0.9), xytext=(0.1, 0.7), arrowprops=dict(arrowstyle='->'), fontsize=10,
#             #                  xycoords='axes fraction', textcoords='axes fraction', ha='center')
#
#         else:
#             axes[0].plot(
#                 self.data[0, fast_inx[0]-150:fast_inx[-1]], self.data[1, fast_inx[0]-150:fast_inx[-1]], "k",
#                 lw=0.5
#             )
#             axes[0].scatter(self.data[0, self.samples_of_pca[0]:self.samples_of_pca[1]],
#                             self.data[1, self.samples_of_pca[0]:self.samples_of_pca[1]],
#                             c="r", lw=1, label=f"first {self.init_motion_smp} samples", alpha=0.6
#                             )
#             r = np.sqrt((self.data[:2, self.samples_of_pca[0]:self.samples_of_pca[1]] ** 2).sum(axis=0)).max()
#             lim = np.sqrt((self.data[:2, fast_inx] ** 2).sum(axis=0)).max() * 1.1
#
#             x = np.cos(np.radians([self.rotation_angle, self.rotation_angle + 180])) * r
#             y = np.sin(np.radians([self.rotation_angle, self.rotation_angle + 180])) * r
#             axes[0].plot(y, x, lw=3, zorder=0, alpha=0.3, label="inital direction")
#             axes[0].set_xlim(-lim, lim)
#             axes[0].set_ylim(-lim, lim)
#             axes[0].set_aspect('equal', adjustable='box')
#             axes[0].legend(loc="upper right")
#             axes[0].annotate('North', xy=(0.1, 0.9), xytext=(0.1, 0.7), arrowprops=dict(arrowstyle='->'), fontsize=10,
#                              xycoords='axes fraction', textcoords='axes fraction', ha='center')
#
#
#
#         axes[1].plot(self.data[3]+self.time_delay, fast,"k",lw=0.5)
#         axes[1].plot(self.data[3], slow,"k",lw=0.5)
#
#         fast_string = f"fast N{int(round(self.rotation_angle,0))}"+r'$^\circ$E'
#         slow_string = f"slow N{int(90+round(self.rotation_angle,0))%360}"+r'$^\circ$E'
#         delay_string = f"{int(self.time_delay*1000)}ms delay time applied"
#
#         axes[1].plot(self.data[3, fast_inx] + self.time_delay, fast[fast_inx],
#                      label=fast_string, color='red')
#         axes[1].plot(self.data[3, slow_inx], slow[slow_inx],label=slow_string, color='green')
#
#
#         axes[1].axvspan(self.data[3, slow_inx[0]], self.data[3, slow_inx[-1]], alpha=0.1, color='green')
#         axes[1].axvspan(self.data[3, fast_inx[0]] + self.time_delay, self.data[3, fast_inx[-1]] + self.time_delay,
#                         alpha=0.1, color='red')
#
#         axes[1].annotate(delay_string, xy=(0, 0), xytext=(0.05, 0.1), fontsize=10,
#                        xycoords='axes fraction', textcoords='axes fraction',
#                        bbox=dict(facecolor='white', alpha=0.8),
#                        horizontalalignment='left', verticalalignment='top')
#
#         axes[1].legend(loc="upper right")
#
#
#
#         axes[2].plot(self.data[3], fast,"k",lw=0.5)
#         axes[2].axvspan(self.data[3, fast_inx[0]], self.data[3, fast_inx[-1]], alpha=0.1, color='red')
#         # axes[2].axhline(y= self.noise_in_fast,c="grey")
#         # axes[2].axhline(y=-self.noise_in_fast, c="grey")
#         axes[2].annotate(fast_string, xy=(0, 0), xytext=(0.05, 0.3), fontsize=10,
#                          xycoords='axes fraction', textcoords='axes fraction',
#                          bbox=dict(facecolor='white', alpha=0.8),
#                          horizontalalignment='left', verticalalignment='top')
#
#         axes[3].plot(self.data[3], slow,"k",lw=0.5)
#         axes[3].axvspan(self.data[3, slow_inx[0]], self.data[3, slow_inx[-1]], alpha=0.1, color='green')
#         axes[3].annotate(slow_string, xy=(0, 0), xytext=(0.05, 0.3), fontsize=10,
#                          xycoords='axes fraction', textcoords='axes fraction',
#                          bbox=dict(facecolor='white', alpha=0.8),
#                          horizontalalignment='left', verticalalignment='top')
#         # self.axes[i].axvspan(self.data[3, inx_s], self.data[3, inx_e], alpha=0.1, color='red')
#         axes[4].plot(self.data[3], self.data[2],"k",lw=0.5)
#         axes[4].annotate("vertical", xy=(0, 0), xytext=(0.05, 0.3), fontsize=10,
#                          xycoords='axes fraction', textcoords='axes fraction',
#                          bbox=dict(facecolor='white', alpha=0.8),
#                          horizontalalignment='left', verticalalignment='top')
#
#
#         y_lim = np.abs(np.concatenate([fast[fast_inx],slow[slow_inx]])).max()*1.2
#         for i in range(1,5):
#             axes[i].set_xlim(self.data[3, slow_inx[0] - 20], self.data[3, slow_inx[-1] + 30])
#             axes[i].set_ylim(-y_lim, y_lim)
#         # axes[2].plot(self.data[3, fast_inx]+self.time_delay, fast[fast_inx])
#         # axes[2].plot(self.data[3, slow_inx], slow[slow_inx])
#         # axes[2].set_xlim(self.data[3, slow_inx[0]], self.data[3, slow_inx[-1]])
#         # axes[2].set_ylim(-y_lim, y_lim)
#
#         axes[4].get_shared_x_axes().join(axes[1], axes[2], axes[3], axes[4])
#         axes[4].get_shared_y_axes().join(axes[1], axes[2], axes[3], axes[4])
#         axes[4].set_xlabel("Time [s]")
#
#
#
#         lim = np.sqrt((self.shifted_data[:] ** 2).sum(axis=0)).max() * 1.1
#         axes[5].plot(self.shifted_data[0],self.shifted_data[1],"k",lw=0.5)#,self.data[2,start_of_slow:end_of_slow])
#         axes[5].set_xlim(-lim, lim)
#         axes[5].set_ylim(-lim, lim)
#         axes[5].set_aspect('equal', adjustable='box')
#         axes[5].set_title("After applying a delay to fast component")
#         # axes[5].plot(
#         #     self.data[0, fast_inx[0]-100:fast_inx[-1]], self.data[1, fast_inx[0]-100:fast_inx[-1]], "k",
#         #     lw=0.5
#         # )  # ,self.data[2,start_of_slow:end_of_slow])
#         # axes[5].set_xlim(-lim, lim)
#         # axes[5].set_ylim(-lim, lim)
#         # axes[5].set_aspect('equal', adjustable='box')
#         # axes[5].set_title("With noise")
#
#         from string import ascii_lowercase
#         for i in range(len(axes)):
#
#             if i in [0,5]:
#                 axes[i].annotate(f'1.{ascii_lowercase[i]}',xy=(0,0),xytext=(0.94,0.03),xycoords='axes fraction',
#                                  textcoords='axes fraction')
#             else:
#                 axes[i].annotate(f'1.{ascii_lowercase[i]}', xy=(0, 0), xytext=(0.94, 0.07), xycoords='axes fraction',
#                                  textcoords='axes fraction')
#             axes[i].set_yticks([])
#             if i==4:
#                 continue
#             axes[i].set_xticks([])
#
#
#
#         fig.subplots_adjust(hspace=0, wspace=0)
#         # fig.subplots_adjust(left=-0.1, hspace=0,wspace=0)
#
#     def determine_time_of_template_by_dist(self):
#         pass
#
#     @classmethod
#     def set_pick_data(cls, path):
#         df = pd.read_csv(path, dtype={"event": str, "pickTime": str})
#         times = np.array(df.pickTime.tolist(), dtype="datetime64[ms]")
#         df["pickTime_floats"] = times.astype("float")
#         df = df.merge(cls.orig_df,left_on="event",right_on="evid")
#         df.loc[df.depth < 5, "depth"] = 5
#         df.loc[df.depth > 30, "depth"] = 30
#         # print(cls.stats)
#         cls.pick_df = df
#
#     @classmethod
#     def set_approx_velocities(cls,v_low,v_high):
#         cls.v_low = v_low
#         cls.v_high = v_high
#
#     @classmethod
#     def set_sites(cls, path):
#         df = pd.read_csv(path)
#         df.rename({"longitude":"lon", "latitude":"lat"}, axis=1, inplace=True)
#         cls.site_df = df
#
#     @classmethod
#     def set_half_cycle_num(cls,num):
#         cls.half_cycle_num = num
#
#     @classmethod
#     def set_first_high_coef_value(cls, val):
#         cls.first_high_coef_val = val

