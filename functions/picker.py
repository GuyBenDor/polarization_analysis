import math

from matplotlib import pyplot as plt
import numpy as np
from matplotlib.widgets import Button, CheckButtons, RadioButtons,MultiCursor,TextBox,Slider, RangeSlider
from matplotlib.gridspec import GridSpec
from matplotlib.dates import date2num, num2date, drange
from matplotlib.transforms import blended_transform_factory
import matplotlib
import os
import pandas as pd
from functions.baseFunctions import find_nearest, mat_rotation, make_array,traces
import threading

matplotlib.rcParams.update(
    {
        "keymap.pan": "z",
        "keymap.save": "p"
    }
)
cursor_colors = ["tab:green", "tab:blue", "tab:red"]
DEBOUNCE_DUR = 0.5
t = None
status_values = {
    0: "N",
    1: "P",
    2: "S"
}

class VerticalLine:

    # global cursor_colors

    def __init__(self, st, **kwargs):
        self.num_check = 0
        self.obj = st
        self.rotAngle = 0
        self.status = 0
        self.pick_level = status_values[self.status]
        self.picking_levels = [self.pick_level] + self.karnelPickNames["P"] + self.karnelPickNames["S"]
        self.kwargs = kwargs


        self.get_picks_in_stream()

        # self.set_originalData()
        self.set_figure(**kwargs)
        self.plot_traces_in_figure(**kwargs)
        self.plot_motion_in_figure()
        self.set_buttons(**kwargs)
        self.set_interaction(**kwargs)

    def set_figure(self, **kwargs):

        self.fig = plt.figure(figsize=(15, 8))
        gs = GridSpec(6, 18, figure=self.fig)

        self.axes = [self.fig.add_subplot(gs[step:step+2, 7:]) for step in range(0,6,2)]
        if 'fig_3d' in kwargs:
            if kwargs['fig_3d']:
                self.axes += [self.fig.add_subplot(gs[:, :7], projection='3d')]
            else:
                self.axes += [self.fig.add_subplot(gs[:, :7])]

        self.saveTextBox = self.axes[2].annotate("data was saved",
                                                 xy=(0, 0), xytext=(0.5, 1.5), fontsize=30, rotation=30,
                                                 xycoords='axes fraction', textcoords='axes fraction',
                                                 bbox=dict(boxstyle="round", facecolor='white', alpha=0.8),
                                                 horizontalalignment='center', verticalalignment='center')

        self.saveTextBox.set_visible(False)

        plt.subplots_adjust(hspace=0)
        plt.suptitle(self.obj.st.stats.event)

    def plot_traces_in_figure(self, **kwargs):

        self.update_data()
        self.ls = []
        self.pickLines = []
        # td = timedelta(milliseconds=1000 / self.obj.st.stats.sampling_rate)
        # self.datetimeRange = drange(
        #     datetime.strptime(str(self.obj.st.stats.starttime), "%Y-%m-%dT%H:%M:%S.%f"),
        #     datetime.strptime(str(self.obj.st.stats.endtime), "%Y-%m-%dT%H:%M:%S.%f")+td,
        #     td
        # )
        for i in range(3):
            # l, = self.axes[i].plot(self.datetimeRange, self.data[i], "k", lw=0.5)
            l, = self.axes[i].plot(self.data[-1].astype("datetime64[ms]"), self.data[i], "k", lw=0.5)
            self.axes[i].axvline(x=self.obj.st.orig_time, color="lime", ls="--")

            # for j in range(len(self.picksInStream.pickTime)):#.values:
            #     # print(str(pick))
            #     # print()
            #     self.axes[i].axvline(x=self.picksInStream.iloc[j].pickTime, color="grey", ls="--", zorder=0, lw=0.8)
                # self.axes[i].axvline(x=datetime.strptime(str(pick)[:-3], "%Y-%m-%dT%H:%M:%S.%f"),color="y")
            self.axes[i].vlines(
                self.picksInStream.pickTime.values, 0, 1, transform=self.axes[i].get_xaxis_transform(), color="grey",
                ls="--", zorder=0, lw=0.8
            )
            temp = []
            for j in range(1,3):
                # lc = LineCollection(
                #     [], color=cursor_colors[j],
                #     transform=self.axes[i].get_xaxis_transform()
                # )
                # self.axes[i].add_collection(lc)
                pl = self.axes[i].vlines(
                    [], 0, 1, transform=self.axes[i].get_xaxis_transform(),
                    color=cursor_colors[j]#,lw=0.8
                )
                temp.append(pl)
            self.pickLines.append(temp)
            self.ls.append(l)
            self.axes[i].annotate(self.obj.st.stats.instrument[:-1] + self.obj.st.stats.columnNames[1:].split(",")[i],
                                  xy=(0, 0), xytext=(0.1, 0.9), fontsize=10,
                                  xycoords='axes fraction', textcoords='axes fraction',
                                  bbox=dict(facecolor='white', alpha=0.8),
                                  horizontalalignment='right', verticalalignment='top')
            if i != 2:
                self.axes[i].set_xticks([])
        # self.picklines = {}
        self.tform = blended_transform_factory(self.axes[2].transData, self.axes[2].transAxes)
        self.annotations = []
        for i in range(len(self.eventPicks_tst)):
            annotation = self.axes[2].annotate(
                        '',
                        xy=(self.eventPicks_tst.iloc[i].d2n, 0.1), xycoords=self.tform,
                        xytext=(self.eventPicks_tst.iloc[i].d2n, 0.1),
                        textcoords=self.tform,bbox=dict(facecolor='white', alpha=0.8),
                                          horizontalalignment='left', verticalalignment='top'
                    )
            self.annotations.append(annotation)

        # for i in range(len(self.eventPicks)):
        #     annotation = self.axes[2].annotate(
        #         int(self.eventPicks.iloc[i].confidence),
        #         xy=(date2num(self.eventPicks.iloc[i].pickTime), 0.1), xycoords=self.tform,
        #         xytext=(date2num(self.eventPicks.iloc[i].pickTime), 0.1),
        #         textcoords=self.tform,bbox=dict(facecolor='white', alpha=0.8),
        #                           horizontalalignment='left', verticalalignment='top'
        #     )
        #     self.annotations.append(annotation)

        self.plot_lineCollection()

        # print(self.eventPicks)
        self.axes[2].get_shared_x_axes().join(self.axes[0], self.axes[1], self.axes[2])
        self.axes[2].get_shared_y_axes().join(self.axes[0], self.axes[1], self.axes[2])

        limlist = [
            self.obj.st.orig_time - np.timedelta64(self.stream_length[0], "s"),
            self.obj.st.orig_time + np.timedelta64(self.stream_length[1], "s")
        ]
        self.axes[2].set_xlim(
            limlist[0],
            limlist[1]
        )
        ilim = np.where(
            (self.obj.st.data[-1].astype("datetime64[ms]") > limlist[0]) &
            (self.obj.st.data[-1].astype("datetime64[ms]") < limlist[1])
        )[0]
        if len(ilim) == 0:
            return
        y_max = np.max(np.abs(self.obj.st.data[:3, ilim])) * 1.2
        self.lsylim = [-y_max, y_max]
        self.axes[2].set_ylim(self.lsylim[0], self.lsylim[1])

    def plot_motion_in_figure(self):
        self.line3d, = self.axes[3].plot([], [], [], "k", lw=0.5)
        self.scatter3d = self.axes[3].scatter([], [], [], c="r", lw=1, alpha=0.6)
        self.rotation_line, = self.axes[3].plot([], [], [], "cyan", alpha=0.7)
        # print(self.eventPicks_tst.pickTime.isnull()[1])
        # if self.eventPicks_tst.Phase.is_null()#[self.eventPicks_tst.Phase=="S"].
        if not self.eventPicks_tst.pickTime.isnull()[1]:
            pickTime = self.eventPicks_tst[self.eventPicks_tst.Phase == "S"].iloc[0].pickTime
            inx = find_nearest(self.data[4].astype("datetime64[ms]"), np.datetime64(pickTime))
            self.update_pm(inx)

        # if "S" in self.eventPicks.Phase.values:
        #     pickTime = self.eventPicks[self.eventPicks.Phase=="S"].iloc[0].pickTime
        #     inx = find_nearest(self.data[4].astype("datetime64[ms]"), np.datetime64(pickTime))
        #     self.particle_motion(inx)
            # print(inx)

    def update_pm(self, inx):

        first_motion = self.filter_data[
                 :3, inx: inx + int(self.time_for_pca * self.obj.st.stats.sampling_rate)
                 ]
        motion = self.filter_data[
                           :3, inx - 100: inx + int((self.time_for_pca + 0.2) * self.obj.st.stats.sampling_rate)
                           ]

        self.figlim = 1.1 * np.sqrt((motion ** 2).sum(axis=0)).max()
        zeros = np.zeros([2, 3])
        zeros[0, 0] = self.figlim * np.sin(np.radians([self.rotAngle]))
        zeros[0, 1] = self.figlim * np.cos(np.radians([self.rotAngle]))

        self.rotation_line.set_data_3d(zeros.T[0], zeros.T[1], zeros.T[2])
        self.line3d.set_data_3d(motion[0], motion[1], motion[2])
        self.axes[3].set_zlim(-self.figlim, self.figlim)
        self.axes[3].set_xlim(-self.figlim, self.figlim)
        self.axes[3].set_ylim(-self.figlim, self.figlim)
        self.scatter3d._offsets3d = (first_motion[0], first_motion[1], first_motion[2])
        self.axes[3].figure.canvas.draw_idle()

    def plot_lineCollection(self):

        self.eventPicks_tst['texts'] = self.eventPicks_tst["confidence"].apply(
            lambda x: "" if math.isnan(x) else int(x))
        for i in range(3):
            for j in range(len(self.eventPicks_tst)):
                self.pickLines[i][j].set_paths(self.eventPicks_tst.iloc[j].lineCol)
                if i == 0:
                    self.annotations[j].set_text(self.eventPicks_tst.iloc[j].texts)
                    self.annotations[j].set_x(self.eventPicks_tst.iloc[j].d2n)
            self.axes[i].figure.canvas.draw_idle()
        # self.fig.canvas.draw_idle()

    def set_buttons(self, **kwargs):
        save_bot_pos = [0.94, 0.25+0.035*len(self.picking_levels), 0.035, 0.035]
        rest_bot_pos = [0.865, 0.91, 0.035, 0.025]
        pick_bot_pos = [0.94, 0.25, 0.035, 0.035*len(self.picking_levels)]
        filt_range_pos = [0.4, 0.89, 0.5, 0.02]
        rotation_range_pos = [0.92, 0.15, 0.02, 0.7]

        axcolor = 'lightgoldenrodyellow'

        self.saveBut = plt.axes(save_bot_pos)
        self.axBut = plt.axes(pick_bot_pos,facecolor=axcolor)
        self.filt_ax = plt.axes(filt_range_pos)
        self.rot_ax = plt.axes(rotation_range_pos)
        self.reset_ax = plt.axes(rest_bot_pos)

        self.filt_rangeSlider = RangeSlider(
            self.filt_ax, "filter", 0.2, 50, valinit=self.orig_filterValues, orientation="horizontal", valstep=0.1
        )

        self.rotation_slider = Slider(
            ax=self.rot_ax, label="rotation", valmin=0, valmax=359, valinit=0, orientation="vertical", valstep=1
        )


        self.picking_buttons = RadioButtons(self.axBut, self.picking_levels)
        for circle,label in zip(self.picking_buttons.circles, self.picking_buttons.labels):
            # circle.set_radius(0.8)
            label.set_fontsize(10)

        self.save_picking_button = Button(self.saveBut, 'Save', hovercolor='0.975')

        self.reset_button = Button(self.reset_ax, 'Reset', hovercolor='0.975')
        self.reset_button.label.set_fontsize(8)

    def set_interaction(self, **kwargs):
        self.t = None
        self.picking = self.picking_buttons.on_clicked(self.set_pick_status)
        self.filter = self.filt_rangeSlider.on_changed(self.update_filter)
        self.rotation = self.rotation_slider.on_changed(self.update_rotation)
        self.multi = MultiCursor(
            self.fig.canvas, (self.axes[0], self.axes[1], self.axes[2]), color=cursor_colors[self.status], lw=1
        )
        self.save = self.save_picking_button.on_clicked(self.save_data_frame)
        self.connection_id = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.c4 = self.fig.canvas.mpl_connect('key_press_event', self.onpresskey)
        self.c5 = self.reset_button.on_clicked(self.reset)
        plt.show()

    def set_pick_status(self, event):
        if event[0] == "N":
            self.status = 0
        elif event[0] == "P":
            self.status = 1
        elif event[0] == "S":
            self.status = 2
        self.pick_level = event
        self.multi.disconnect()
        self.multi = MultiCursor(self.fig.canvas, (self.axes[0], self.axes[1], self.axes[2]),
                                 color=cursor_colors[self.status], lw=1)
        plt.draw()

    def onpresskey(self, event):
        if event.key == '§' or (event.key == 'z' and self.picking_buttons.get_active != 0):
            self.picking_buttons.set_active(0)
        elif event.key == "cmd+s":
            self.save_data_frame(event)
        elif event.key.isdigit() and int(event.key) in range(1, len(self.picking_levels)):
            self.picking_buttons.set_active(int(event.key))

    def get_picks_in_stream(self):
        # print(len(self.pick_df))
        # print(self.pick_df.dtypes)
        self.picksInStream = self.pick_df[
            (self.pick_df.pickTime >= np.datetime64(self.obj.st.stats.starttime)) &
            (self.pick_df.pickTime <= np.datetime64(self.obj.st.stats.endtime))
            ]
        # del pick_df
        self.update_eventPicks()

    def save_data_frame(self, event):


        # if self.status != 0:
        #     print("Need to switch to N mode.")
        #     return
        if not hasattr(self, "save_path"):
            print("define a save_path.")
            return
        print(f"Dataframe had {len(self.pick_df)} picks.")
        self.eventPicks_tst["pickTime"] = self.eventPicks_tst["pickTime"].astype("datetime64[ns]")
        self.pick_df = self.pick_df[self.pick_df.evid != self.obj.st.stats.event]
        self.pick_df = pd.concat([
            self.pick_df, self.eventPicks_tst[self.eventPicks_tst['pickTime'].notna()][
                ['plt_num', 'pickTime', 'Phase', 'evid', 'confidence', 'station', 'status']]
        ], ignore_index=True)
        self.pick_df.sort_values(by="plt_num", ignore_index=True, inplace=True)


        print(f"Data was saved.\tNow contains {len(self.pick_df)} picks.")
        self.pick_df.drop(["station", "status"], axis=1).to_csv(
            os.path.join(self.save_path, f'{self.obj.st.stats.station}.csv'), index=False
        )
        self.update_picksOfStation(self.pick_df)

        self.saveTextBox.set_visible(True)
        self.fig.canvas.draw_idle()
        threading.Timer(1, self.remove_notify_dataSave).start()
        # self.axes[2].figure.canvas.draw_idle()

    def reset(self, event):
        self.filt_rangeSlider.set_val(self.orig_filterValues)
        # self.filt_rangeSlider.reset()

    def remove_notify_dataSave(self):
        self.saveTextBox.set_visible(False)
        self.axes[2].figure.canvas.draw_idle()

    def onclick(self, event):
        if not any([0 if event.inaxes != ax else 1 for ax in self.axes[:-1]]): return
        # self.num_check += 1
        # print(self.num_check)
        if self.t is None:
            self.t = threading.Timer(DEBOUNCE_DUR, self.on_singleclick, [event])
            self.t.start()
        if event.dblclick:
            self.t.cancel()
            self.on_dblclick(event)

    def handle_existing_pick(self):
        row_to_change = self.eventPicks[self.eventPicks.status == self.status]
        self.eventPicks.drop(index=row_to_change.index.values, inplace=True)
        self.eventPicks.sort_values(by="plt_num", ignore_index=True, inplace=True)

    def on_singleclick(self, event):
        if self.status != 0 and not event.button == 3:
            ser = {
                'plt_num': event.xdata,
                'pickTime': num2date(event.xdata).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3],
                'confidence': int(self.pick_level.split(' ')[1])
            }
            ser.update({"d2n": date2num([ser["pickTime"]])[0]})
            ser.update({'lineCol': [np.array([[[ser["d2n"], 1], [ser["d2n"], -1]]])]})
            self.eventPicks_tst.loc[
                self.eventPicks_tst[self.eventPicks_tst.status == self.status].index,
                ['plt_num', 'pickTime', 'confidence', "d2n", "lineCol"]] = ser.values()
            self.plot_lineCollection()
            r = 3
            if self.status == 2:
                inx = find_nearest(self.data[4].astype("datetime64[ms]"), np.datetime64(ser['pickTime']))
                self.update_pm(inx)

            #     r = 4
            # for i in range(r):
            #     self.axes[i].figure.canvas.draw_idle()
        self.t = None

    def on_dblclick(self, event):
        if not self.eventPicks_tst.pickTime.isnull().values.all():
            min_max = self.axes[0].get_xlim()
            linex = [lc.get_paths()[0].vertices[0, 0] if lc.get_paths() else np.nan for lc in
                     self.axes[0].collections[1:]]
            relative_err = np.absolute(event.xdata - np.array(linex)) / (min_max[1] - min_max[0])
            inx = np.nanargmin(relative_err)

            if relative_err[inx] < 0.005:
                print(f"A pick of type {self.eventPicks_tst.iloc[inx].Phase} was deleted")
                self.eventPicks_tst.loc[
                    inx, ['plt_num', 'pickTime', 'confidence', 'd2n']
                ] = [np.nan, np.nan, np.nan, np.nan]
                self.eventPicks_tst["lineCol"] = self.eventPicks_tst["d2n"].apply(make_array)
                self.plot_lineCollection()
            for i in range(3):
                self.axes[i].figure.canvas.draw_idle()
        self.t = None

    def update_data(self):
        dtimes, times = self.obj.st.data[3:, :]
        b = dtimes[:, None]
        c = times[:, None]
        chan_labels = "ENZ"
        stream = self.obj.st.stream.copy().filter(
            "bandpass", freqmin=self.filterValues[0], freqmax=self.filterValues[1], corners=4
        )
        arr = np.array([stream.select(channel="??" + ch)[0].data for ch in chan_labels])
        self.filter_data = np.hstack((arr.copy().T, b, c)).T
        arr = mat_rotation(self.rotAngle, arr)
        self.data = np.hstack((arr.T, b, c)).T  # [E,N,Z, delta_times, times]

    def update_eventPicks(self):
        self.eventPicks_tst = pd.DataFrame(index=np.arange(2), columns=["station", "evid", "Phase", "status"])
        self.eventPicks_tst[["Phase", "status"]] = [["P", 1], ["S", 2]]
        self.eventPicks_tst[["evid", "station"]] = [self.obj.st.stats.event, self.obj.st.stats.station]
        self.eventPicks_tst = self.eventPicks_tst.merge(self.pick_df, how="left",
                                                        on=["station", "evid", "Phase", "status"])
        self.eventPicks_tst["d2n"] = date2num(self.eventPicks_tst["pickTime"].values)
        self.eventPicks_tst["lineCol"] = self.eventPicks_tst["d2n"].apply(make_array)

    def update_filter(self, val):
        # print(val)
        self.filterValues = val
        # print(f"bandpass {val[0]}-{val[1]} Hz filter applied")
        self.update_data()
        for i in range(3):
            self.ls[i].set_ydata(self.data[i])
        if not self.eventPicks_tst.pickTime.isnull()[1]:
            pickTime = self.eventPicks_tst[self.eventPicks_tst.Phase == "S"].iloc[0].pickTime
            inx = find_nearest(self.data[4].astype("datetime64[ms]"), np.datetime64(pickTime))
            self.update_pm(inx)
        # self.plot_motion_in_figure()
        # self.fig.canvas.draw_idle()

    def update_rotation(self, val):
        self.rotAngle = val
        # print(f"rotation of {val} applied")
        self.update_data()
        for i in range(3):
            self.ls[i].set_ydata(self.data[i])


        if hasattr(self,"figlim"):
            zeros = np.zeros([2, 3])
            zeros[0, 0] = self.figlim * np.sin(np.radians([self.rotAngle]))
            zeros[0, 1] = self.figlim * np.cos(np.radians([self.rotAngle]))

            self.rotation_line.set_data_3d(zeros.T[0], zeros.T[1], zeros.T[2])


        self.fig.canvas.draw_idle()

    @classmethod
    def set_karnelPickNames(cls,Pvalues, Svalues):
        cls.karnelPickNames = {
            "P": [f'P {num}' for num in range(Pvalues)],
            "S": [f'S {num}' for num in range(Svalues)]
        }

    @classmethod
    def set_filterValues(cls,minMax):
        cls.orig_filterValues = cls.filterValues = minMax

    @classmethod
    def set_time_for_pca(cls, time):
        cls.time_for_pca = time

    @classmethod
    def set_stream_length(cls, times):
        cls.stream_length = times

    @classmethod
    def set_save_path(cls, save_path):
        cls.save_path = save_path

    @classmethod
    def get_picksOfStation(cls, station):
        if not hasattr(traces, "pick_df"):
            print("Set a pick_df first")
            return
        df = traces.pick_df[traces.pick_df.station == station].copy()
        if len(df) != 0:
            df.reset_index(inplace=True, drop=True)
        if "confidence" not in df.columns:
            df['confidence'] = np.nan
        if "status" not in df.columns:
            df['status'] = df.Phase.map({"P": 1, "S": 2})
        df['evid'] = df['evid'].astype(str)
        # print(df.dtypes)
        # print(len(df))
        cls.pick_df = df

    @classmethod
    def update_picksOfStation(cls, df):
        cls.pick_df = df

class Figure:

    def __init__(self, path, save_path, fig_3d=False, Type=None):

        self.st = traces(path,Type=Type)
        self.save_path = os.path.join(save_path, self.st.stats.station + ".csv")
        # self.pick_df = self.get_picks(save_path)
        self.fc = VerticalLine(self, fig_3d=fig_3d)

    # def get_picks(self, save_path):
    #
    #     if os.path.exists(save_path):
    #         return pd.read_csv(save_path, dtype={"event": str})
    #     else:
    #         return pd.DataFrame(columns=["plt_num", "pickTime", "Phase", "event"])

    @classmethod
    def set_origins(cls,path):
        traces.set_origins(path)

    @classmethod
    def set_filter_values(cls,vals):
        VerticalLine.set_filterValues(vals)
        traces.set_filter_limits(vals[0], vals[1])

    @classmethod
    def set_time_for_pca(cls, time):
        VerticalLine.set_time_for_pca(time)

    @classmethod
    def set_stream_length(cls,times):
        VerticalLine.set_stream_length(times)

    @classmethod
    def set_pick_path(cls, pick_path):
        traces.set_picks(pick_path)

    @classmethod
    def set_karnelPickNames(cls, pValues, sValues):
        VerticalLine.set_karnelPickNames(pValues,sValues)

    @classmethod
    def set_save_path(cls, save_path):
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        VerticalLine.set_save_path(save_path)

    @classmethod
    def set_station(cls, station):
        VerticalLine.get_picksOfStation(station)