import os
import pandas as pd
from functions.picker import Figure,VerticalLine
from functions.baseFunctions import angle, initial_bearing, haversine,mat_rotation, find_nearest
import math
import numpy as np
from matplotlib.backend_tools import Cursors
import matplotlib.pyplot as plt
from matplotlib.backend_bases import MouseButton
from matplotlib.gridspec import GridSpec
import obspy
from matplotlib.transforms import blended_transform_factory
from matplotlib.dates import date2num, num2date, drange
from matplotlib.widgets import Button, CheckButtons, RadioButtons,MultiCursor,TextBox,Slider, RangeSlider
import threading
import matplotlib

if 'cmd+w' in matplotlib.rcParams['keymap.quit']:
    matplotlib.rcParams['keymap.quit'].remove('cmd+w')
matplotlib.rcParams.update(
    {
        "keymap.pan": "z",
        "keymap.save": "p"
    }
)

Rn = [2, 1, 0]
picklines = {
    "P": "tab:blue", "S": "tab:red"
}

cursor_colors = ["tab:green", "tab:blue", "tab:red"]
DEBOUNCE_DUR = 0.5
t = None
status_values = {
    0: "N",
    1: "P",
    2: "S"
}

key_list = list(status_values.keys())
val_list = list(status_values.values())

def multiStation_fig(fig, streams, evid, pick_df):
    plt.clf()
    axes = []
    num = len(streams.keys())
    r, step = np.linspace(0.75, -0.05, num=num + 1, retstep=True)
    pos = [0.05, 0.36, 0.67]
    for i, [key, val] in enumerate(streams.items()):
        picks = pick_df[pick_df.station==key]
        maxi = np.sqrt((val.data[:3] ** 2).sum(axis=0)).max()
        temp = []
        for j, p in enumerate(pos):
            ax = fig.add_axes([p, r[i], 0.3, -step])
            ax.plot(val.data[-1].astype("datetime64[ms]"), val.data[j] / maxi, "k", lw=0.5)
            ax.set_xlim(val.orig_time - np.timedelta64(1, "s"),
                        val.orig_time + np.timedelta64(30, "s"))

            ax.axvline(x=val.orig_time, color="lime", ls="--")
            pl = ax.vlines(
                picks.pickTime.values, 0, 1,
                transform=ax.get_xaxis_transform(),
                color=picks.color.values

            )

            if i == 0:
                ax.set_title(f'EN{val.stats.columnNames[j]}', fontsize=16)
            temp.append(ax)

        tform = blended_transform_factory(temp[2].transData, temp[2].transAxes)
        for i in range(len(picks)):
            row = picks.iloc[i]
            annot = temp[2].annotate(
                int(row.confidence) if not np.isnan(row.confidence) else "",
                xy=(date2num(row.pickTime), 0.1), xycoords=tform,
                xytext=(date2num(row.pickTime), 0.25),
                textcoords=tform, bbox=dict(facecolor='white', alpha=0.8),
                horizontalalignment='left', verticalalignment='top'
            )

        temp[-1].annotate(
            key, xy=(0, 0), xytext=(0.9, 0.8), fontsize=8, xycoords='axes fraction',
            textcoords='axes fraction',
            bbox=dict(facecolor='white', alpha=0.8), horizontalalignment='right', verticalalignment='top'
        )
        temp[0].get_shared_x_axes().join(temp[0], *temp[1:])
        temp[0].get_shared_y_axes().join(temp[0], *temp[1:])

        axes.append(temp)
    axes = np.array(axes)
    axes[0, 0].get_shared_x_axes().join(axes[0, 0], *axes[1:, 0])
    axes[0, 0].get_shared_y_axes().join(axes[0, 0], *axes[1:, 0])
    fig.suptitle(evid, fontsize='large')
    return fig, axes

class singleStation_picker:

    def __init__(self, sta, fig, picks_df):

        self.st = self.st_dic[sta]
        self.rotAngle = 0
        self.status = 0
        self.pca_pick_view = "S"
        self.pca_extent = self.pca_extent_init
        self.pick_level = status_values[self.status]
        self.picking_levels = [self.pick_level] + self.karnelPickNames["P"] + self.karnelPickNames["S"]
        self.staData = self.st_df[self.st_df.station == sta]
        self.filterValues = self.staData[['fmin','fmax']].values.tolist()[0]
        self.station = sta
        self.picks = picks_df.copy()
        self.pl = []
        self.annotations = []
        self.ls = []

        self.evPicks = self.picks.copy()[
            (self.picks.station == sta) &
            (self.picks.evid == self.st.stats.event)
            ]
        # ser.update({"d2n": date2num([ser["pickTime"]])[0]})
        self.evPicks["d2n"] = self.evPicks.pickTime.apply(date2num)
        self.evPicks.reset_index(drop=True, inplace=True)
        # self.filter_values = self.st_df[self.st_df.station == sta][["fmin","fmax"]].iloc[0].values
        self.fig = fig
        self.working_data()
        self.make_fig()

    def working_data(self):
        dtimes, times = self.st.data[3:, :]
        b = dtimes[:, None]
        c = times[:, None]
        chan_labels = "ENZ"
        stream = self.st.stream.copy().filter(
            "bandpass", freqmin=self.filterValues[0], freqmax=self.filterValues[1], corners=4
        )
        arr = np.array([stream.select(channel="??" + ch)[0].data for ch in chan_labels])
        self.filter_data = np.hstack((arr.copy().T, b, c)).T
        arr = mat_rotation(self.rotAngle, arr)
        self.data = np.hstack((arr.T, b, c)).T  # [E,N,Z, delta_times, times]

    def make_fig(self):

        r, step = np.linspace(0.6, -0.05, num=4, retstep=True)
        axes = []
        for i in range(3):
            axes.append(self.fig.add_axes([0.5, r[i], 0.43, -step]))

        axes[0].get_shared_x_axes().join(axes[0], *axes[1:])
        axes[0].get_shared_y_axes().join(axes[0], *axes[1:])
        axes.append(self.fig.add_axes([0, 0.1, 0.45, 0.9], projection='3d'))
        axes[-1].view_init(elev=90., azim=270)
        self.axes = np.array(axes)

        self.saveTextBox = self.axes[2].annotate("data was saved",
                                                 xy=(0, 0), xytext=(0.5, 1.5), fontsize=30, rotation=30,
                                                 xycoords='axes fraction', textcoords='axes fraction',
                                                 bbox=dict(boxstyle="round", facecolor='white', alpha=0.8),
                                                 horizontalalignment='center', verticalalignment='center')

        self.saveTextBox.set_visible(False)

        self.plot_traces()
        self.add_plot_settings()
        self.plot_picks()
        self.plot_motion()
        self.set_buttons()
        self.set_interaction()
        # print(self.evPicks.columns)

    def plot_traces(self):
        for i, tr in enumerate(self.data[:3]):
            self.ls.append(self.axes[i].plot(self.data[-1].astype("datetime64[ms]"), tr, "k", lw=0.5))

    def plot_motion(self):
        self.line3d, = self.axes[3].plot([], [], [], "k", lw=0.5)
        self.scatter3d = self.axes[3].scatter([], [], [], c="r", lw=1, alpha=0.6)
        self.rotation_line, = self.axes[3].plot([], [], [], "cyan", alpha=0.7)
        # print(self.eventPicks_tst.pickTime.isnull()[1])
        # if self.eventPicks_tst.Phase.is_null()#[self.eventPicks_tst.Phase=="S"].
        self.spans = []
        self.phase_view(self.pca_pick_view)

    def add_plot_settings(self):
        self.tform = blended_transform_factory(self.axes[2].transData, self.axes[2].transAxes)
        for i, tr in enumerate(self.data[:3]):
            self.axes[i].axvline(x=self.st.orig_time, color="lime", ls="--")

            # if i == 0:
            #     self.annotations[j].set_text(self.eventPicks_tst.iloc[j].texts)
            #     self.annotations[j].set_x(self.eventPicks_tst.iloc[j].d2n)
            self.axes[i].annotate(self.st.stats.instrument[:-1] + self.st.stats.columnNames[i],
                                  xy=(0, 0), xytext=(0.1, 0.9), fontsize=10,
                                  xycoords='axes fraction', textcoords='axes fraction',
                                  bbox=dict(facecolor='white', alpha=0.8),
                                  horizontalalignment='right', verticalalignment='top')
            if i != 2:
                self.axes[i].set_xticks([])

        self.axes[0].set_xlim(self.st.orig_time - np.timedelta64(1, "s"), self.st.orig_time + np.timedelta64(10, "s"))
        self.fig.suptitle(
            f'{self.st.stats.station} - {self.st.stats.event}\ndistance: {round(self.staData.dist.values[0],2)}',
            fontsize='large', y=0.9
        )

    def plot_picks(self):
        for i in reversed(range(len(self.pl))):
            self.pl[i].remove()
            self.pl.pop(i)

        self.pl = [ax.vlines(
            self.evPicks.pickTime.values, 0, 1,
            transform=ax.get_xaxis_transform(),
            color=self.evPicks.color.values
        ) for ax in self.axes[:3]]

        for i in reversed(range(len(self.annotations))):
            self.annotations[i].remove()
            self.annotations.pop(i)

        for i in range(len(self.evPicks)):
            row = self.evPicks.iloc[i]
            annot = self.axes[2].annotate(
                    int(row.confidence) if not np.isnan(row.confidence) else "",
                    xy=(row.d2n, 0.1), xycoords=self.tform,
                    xytext=(row.d2n, 0.1),
                    textcoords=self.tform,bbox=dict(facecolor='white', alpha=0.8),
                                      horizontalalignment='left', verticalalignment='top'
                )
            self.annotations.append(annot)
        plt.draw()

    def update_motion(self, inx):
        self.update_span(inx)

        first_motion = self.filter_data[
                       :3, inx: inx + int(self.time_for_pca * self.st.stats.sampling_rate)
                       ]
        motion = self.filter_data[
                 :3, inx + int(self.st.stats.sampling_rate * self.pca_extent[0]):
                     inx + int((self.time_for_pca + self.pca_extent[1]) * self.st.stats.sampling_rate)
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

    def set_buttons(self):
        save_bot_pos = [0.95, 0.25+0.035*len(self.picking_levels), 0.035, 0.035]
        pick_bot_pos = [0.95, 0.25, 0.035, 0.035*len(self.picking_levels)]
        rotation_range_pos = [0.93, 0.15, 0.02, 0.7]

        pca_range_pos = [0.15, 0.04, 0.7, 0.02]
        filt_range_pos = [0.15, 0.01, 0.7, 0.02]  # [0.4, 0.89, 0.5, 0.02]

        rest_bot_pos_ex = [0.05, 0.04, 0.035, 0.02]
        rest_bot_pos = [0.05, 0.01, 0.035, 0.02]

        pca_view_pos = [0.05, 0.07, 0.035, 0.06]

        axcolor = 'lightgoldenrodyellow'

        self.saveBut = plt.axes(save_bot_pos)
        self.axBut = plt.axes(pick_bot_pos,facecolor=axcolor)
        self.filt_ax = plt.axes(filt_range_pos)
        self.rot_ax = plt.axes(rotation_range_pos)
        self.reset_ax = plt.axes(rest_bot_pos)
        self.reset_ax_extent = plt.axes(rest_bot_pos_ex)
        self.pca_range_ax = plt.axes(pca_range_pos)
        self.pca_view_ax = plt.axes(pca_view_pos,facecolor=axcolor)
        # self.pca_view_ax_title = plt.axes([0.05, 0.13, 0.035, 0.03],facecolor=axcolor)
        self.pca_view_ax.text(-0.04,1.1,"pm view")

        self.filt_rangeSlider = RangeSlider(
            self.filt_ax, "filter", 0.2, 50, valinit=self.filterValues, orientation="horizontal", valstep=0.1
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

        self.reset_button1 = Button(self.reset_ax_extent, 'Reset', hovercolor='0.975')
        self.reset_button1.label.set_fontsize(8)

        self.pca_rangeSlider = RangeSlider(
            self.pca_range_ax, "pca time", -2, 2,
            valinit=self.pca_extent_init, orientation="horizontal",
            valstep=1/self.st.stats.sampling_rate
        )

        self.pca_view_buttons = RadioButtons(self.pca_view_ax, ["  P", "  S"],active=1)
        for circle, label in zip(self.pca_view_buttons.circles, self.pca_view_buttons.labels):
            circle.set_radius(0.1)
            label.set_fontsize(10)

    def set_interaction(self):
        self.t = None
        self.picking = self.picking_buttons.on_clicked(self.set_pick_status)
        self.filter = self.filt_rangeSlider.on_changed(self.update_filter)
        self.pca_range_extent = self.pca_rangeSlider.on_changed(self.update_pca_extent)
        self.rotation = self.rotation_slider.on_changed(self.update_rotation)
        self.multi = MultiCursor(
            self.fig.canvas, (self.axes[0], self.axes[1], self.axes[2]), color=cursor_colors[self.status], lw=1
        )
        self.save = self.save_picking_button.on_clicked(self.save_data_frame)
        self.connection_id = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.c4 = self.fig.canvas.mpl_connect('key_press_event', self.onpresskey)
        self.c5 = self.reset_button.on_clicked(self.reset)
        self.c6 = self.reset_button1.on_clicked(self.reset_extent)
        self.c7 = self.pca_view_buttons.on_clicked(self.phase_view)

    def update_filter(self, val):
        # print(val)
        self.filterValues = val
        # print(f"bandpass {val[0]}-{val[1]} Hz filter applied")
        self.working_data()
        for i in range(3):
            self.ls[i][0].set_ydata(self.data[i])
        self.phase_view(self.pca_pick_view)
        # if not self.evPicks:
        #     pickTime = self.eventPicks_tst[self.eventPicks_tst.Phase == self.pca_pick_view].iloc[0].pickTime
        #     inx = find_nearest(self.data[4].astype("datetime64[ms]"), np.datetime64(pickTime))
        #     self.update_pm(inx)
        # self.plot_motion_in_figure()
        # self.fig.canvas.draw_idle()

    def update_span(self, inx):
        lower_lim = self.data[-1, inx + int(self.st.stats.sampling_rate * self.pca_extent[0])]
        upper_lim = self.data[
            -1, inx + int(self.st.stats.sampling_rate * (self.time_for_pca + self.pca_extent[1]))]
        if len(self.spans) == 0:
            pass
        else:
            for s in self.spans[::-1]:
                s.remove()
                self.spans.remove(s)
        for i in range(3):
            span = self.axes[i].axvspan(
                lower_lim.astype("datetime64[ms]"),
                upper_lim.astype("datetime64[ms]"),
                alpha=0.1, color='green'
            )
            self.spans.append(span)

    def update_pca_extent(self, val):
        self.pca_extent = val
        self.phase_view(self.pca_pick_view)

    def update_rotation(self, val):
        self.rotAngle = val
        # print(f"rotation of {val} applied")
        self.working_data()
        for i in range(3):
            self.ls[i][0].set_ydata(self.data[i])


        if hasattr(self,"figlim"):
            zeros = np.zeros([2, 3])
            zeros[0, 0] = self.figlim * np.sin(np.radians([self.rotAngle]))
            zeros[0, 1] = self.figlim * np.cos(np.radians([self.rotAngle]))

            self.rotation_line.set_data_3d(zeros.T[0], zeros.T[1], zeros.T[2])

        self.fig.canvas.draw_idle()

    def phase_view(self, event):
        self.pca_pick_view = event.strip()
        temp = self.evPicks[self.evPicks.Phase==self.pca_pick_view]
        if len(temp):
            pickTime = temp.iloc[0].pickTime
            inx = find_nearest(self.data[4].astype("datetime64[ms]"), np.datetime64(pickTime))
            self.update_motion(inx)
        plt.draw()

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

    def onclick(self, event):
        if self.fig.canvas.widgetlock.locked(): return
        if not any([0 if event.inaxes != ax else 1 for ax in self.axes[:-1]]): return
        # self.num_check += 1
        # print(self.num_check)
        if self.t is None:
            self.t = threading.Timer(DEBOUNCE_DUR, self.on_singleclick, [event])
            self.t.start()
        if event.dblclick:
            self.t.cancel()
            self.on_dblclick(event)

    def on_singleclick(self, event):
        if self.status != 0 and not event.button == 3:
            ser = {
                'plt_num': event.xdata,
                'pickTime': num2date(event.xdata).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3],
                'confidence': int(self.pick_level.split(' ')[1]),
                'Phase': status_values[self.status],
                'evid': self.st.stats.event,
                'station': self.station,
                "color": picklines[status_values[self.status]]
            }
            ser.update({"d2n": date2num([ser["pickTime"]])})
            if status_values[self.status] in self.evPicks.Phase.values:
                self.evPicks = self.evPicks[self.evPicks.Phase!=status_values[self.status]]
            self.evPicks = pd.concat([self.evPicks, pd.DataFrame(ser)])
            # self.evPicks.append(ser, ignore_index=True)
            self.evPicks.sort_values(by="plt_num", ignore_index=True, inplace=True)
            # print(self.evPicks)

            self.plot_picks()

            if self.pca_pick_view == status_values[self.status]:
                self.phase_view(self.pca_pick_view)

        self.t = None

    def on_dblclick(self, event):
        if len(self.evPicks)>0:
            min_max = self.axes[0].get_xlim()
            ts = [cl.vertices[0, 0] for cl in self.pl[0].get_paths()]
            relative_err = np.absolute(event.xdata - np.array(ts)) / (min_max[1] - min_max[0])
            inx = np.nanargmin(relative_err)

            if relative_err[inx] < 0.005:
                print(f"      A pick of type {self.evPicks.iloc[inx].Phase} was deleted")
                self.evPicks.drop(index=inx, inplace=True)
                self.evPicks.reset_index(drop=True, inplace=True)
                self.plot_picks()
        self.t = None

    def onpresskey(self, event):
        if event.key == '§' or (event.key == 'z' and self.picking_buttons.get_active != 0):
            self.picking_buttons.set_active(0)
        elif (event.key == "cmd+s") or (event.key == "ctrl+s"):
            self.save_data_frame(event)
        elif event.key.isdigit() and int(event.key) in range(1, len(self.picking_levels)):
            self.picking_buttons.set_active(int(event.key))
        elif (event.key == "cmd+w") or (event.key == "ctrl+w"):
            self.pca_view_buttons.set_active(0)
        elif (event.key == "cmd+e") or (event.key == "ctrl+e"):
            self.pca_view_buttons.set_active(1)
        elif event.key == "d":
            self.pca_extent = self.pca_extent_init
            self.pca_rangeSlider.set_val(self.pca_extent)

    def save_data_frame(self, event):

        print(f"   {self.station} had {len(self.picks[self.picks.station==self.station])} picks.")
        self.evPicks["pickTime"] = self.evPicks["pickTime"].astype("datetime64[ns]")
        self.picks = self.picks[~((self.picks.evid == self.st.stats.event) & (self.picks.station == self.station))]
        self.picks = pd.concat([
            self.picks, self.evPicks[self.picks.columns.values]
        ], ignore_index=True)
        self.picks.sort_values(by=["station", "plt_num"], ignore_index=True, inplace=True)
        print(f"   Data was saved.\tNow contains {len(self.picks[self.picks.station==self.station])} picks.")

        self.picks[['plt_num', 'pickTime', 'Phase', 'evid', 'confidence', 'station']].to_csv(
            self.save_path, index=False
        )

        self.picks[self.picks.station == self.station][[
            'plt_num', 'pickTime', 'Phase', 'evid', 'confidence', 'station'
        ]].to_csv(
            f'{os.path.join(self.single_savePath,self.station)}.csv', index=False
        )

        picker.update_pickDf(self.picks)

        self.saveTextBox.set_visible(True)
        self.fig.canvas.draw_idle()
        threading.Timer(1, self.remove_notify_dataSave).start()
        # self.axes[2].figure.canvas.draw_idle()

    def reset(self, event):
        self.filt_rangeSlider.set_val(self.staData[['fmin','fmax']].values.tolist()[0])
        # self.filt_rangeSlider.reset()

    def reset_extent(self, event):
        self.pca_rangeSlider.set_val(self.pca_extent_init)

    def remove_notify_dataSave(self):
        self.saveTextBox.set_visible(False)
        self.axes[2].figure.canvas.draw_idle()

    @classmethod
    def set_stations(cls, st_dic):
        cls.st_dic = st_dic

    @classmethod
    def set_event(cls, evDa):
        cls.evDa = evDa

    @classmethod
    def station_info(cls,st_df):
        cls.st_df = st_df

    @classmethod
    def set_karnelPickNames(cls, Pvalues, Svalues):
        cls.karnelPickNames = {
            "P": [f'P {num}' for num in range(Pvalues)],
            "S": [f'S {num}' for num in range(Svalues)]
        }

    @classmethod
    def set_time_for_pca(cls, time):
        cls.time_for_pca = time

    @classmethod
    def set_stream_length(cls, times):
        cls.stream_length = times

    @classmethod
    def set_pca_extent(cls, Min, Max):
        cls.pca_extent_init = [-Min, Max]

    @classmethod
    def set_save_path(cls, save_path):
        cls.save_path = save_path

    @classmethod
    def set_single_station_path(cls, path):
        cls.single_savePath = path


class picker:

    def __init__(self, evid):
        self.evid = evid
        self.stream = {}

        self.evData = self.orig[self.orig.evid == evid].iloc[0]
        self.sort_by_distance()
        self.setClick = False
        self.Type = "ms"

        for i in range(len(self.local_station)):
            row = self.local_station.iloc[i]
            traces.set_filter_limits(row.fmin, row.fmax)

            path = os.path.join(self.tracePath, row.station, row.instrument, evid+".mseed")
            if os.path.exists(path):
                self.stream[row.station] = traces(path)
                # {
                #     "st": traces(path), "picks": self.pick_df[
                #         (self.pick_df.station == row.station) & (self.pick_df.evid == evid)
                #     ]
                # }
        # self.get_picks()
        singleStation_picker.station_info(self.local_station)
        singleStation_picker.set_stations(self.stream)
        singleStation_picker.set_event(self.evData)
        self.returnMultiSt()

    def get_picks(self):
        self.local_picks = self.pick_df[
            (self.pick_df.station.isin(self.local_station.station)) &
            (self.pick_df.evid == self.evid)
        ]

    def set_interaction(self):

        c1 = self.fig.canvas.mpl_connect('axes_enter_event', self.enter_axes)
        c2 = self.fig.canvas.mpl_connect('motion_notify_event', self.hover)
        c3 = self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        c4 = self.fig.canvas.mpl_connect('key_press_event', self.onpresskey)
        self.connections = [c1, c2, c3, c4]

    def returnMultiSt(self):
        plt.close()
        self.set_fig()
        self.get_picks()
        self.fig, self.axes = multiStation_fig(self.fig, self.stream, self.evid, self.local_picks)
        self.saveTextBox = self.axes[2,-1].annotate("data was saved",
                                                 xy=(0, 0), xytext=(0.5, 0.6), fontsize=30, rotation=30,
                                                 xycoords='figure fraction', textcoords='figure fraction',
                                                 bbox=dict(boxstyle="round", facecolor='white', alpha=0.8),
                                                 horizontalalignment='center', verticalalignment='center')
        self.saveTextBox.set_visible(False)
        self.set_interaction()
        plt.show()

    def set_single_interaction(self):
        self.temp.fig.canvas.mpl_connect('key_press_event', self.onpresskey)

    def onpresskey(self, event):
        if (event.key == 'cmd+a'):
            self.Type = "ms"
            self.returnMultiSt()
        elif self.Type == 'ms' and event.key == 'cmd+s':
            self.save_data_frame(event)
    #
    # def onpresskey(self, event):


    def hover(self, event):
        self.check = [ax == event.inaxes for ax in self.axes[:, 2]]
        if any(self.check):
            self.station = np.array(list(self.stream.keys()))[self.check][0]
            self.setClick = True
        else:
            self.setClick = False

    def set_fig(self):
        self.fig = plt.figure(figsize=[15, 9.5])

    def enter_axes(self, event):
        if self.fig.canvas.widgetlock.locked():
            return
        if not event.inaxes:
            return

    def on_click(self, event):
        if self.fig.canvas.widgetlock.locked(): return
        if event.button is MouseButton.LEFT and self.setClick:
            plt.close()
            self.set_fig()
            self.temp = singleStation_picker(
                self.station, self.fig, self.pick_df
            )
            self.Type = 'ss'
            self.set_single_interaction()
            plt.show()

    def sort_by_distance(self):
        df = self.st_dic.copy().merge(self.site,left_index=True, right_on="station", how="left")
        df['dist'] = df[["longitude", "latitude"]].apply(
            lambda x: haversine(x[0], x[1], self.evData.lon, self.evData.lat), axis=1
        )
        self.local_station = df.sort_values(by="dist").reset_index(drop=True)

    def save_data_frame(self, event):

        print(f"   Dataframe has {len(self.pick_df)} picks.")
        self.pick_df.sort_values(by=["station", "plt_num"], ignore_index=True, inplace=True)
        self.pick_df[['plt_num', 'pickTime', 'Phase', 'evid', 'confidence', 'station']].to_csv(
            self.save_path, index=False
        )
        for i, g in self.pick_df.groupby('station'):
            g[['plt_num', 'pickTime', 'Phase', 'evid', 'confidence', 'station']].to_csv(
                f'{os.path.join(self.df_savePath, i)}.csv', index=False
            )
            print(f"   {i}    {len(g)}")


        # for station in self.pick_df.station.unique
        picker.update_pickDf(self.pick_df)

        self.saveTextBox.set_visible(True)
        self.fig.canvas.draw_idle()
        threading.Timer(1, self.remove_notify_dataSave).start()
        # self.axes[2].figure.canvas.draw_idle()

    def remove_notify_dataSave(self):
        self.saveTextBox.set_visible(False)
        self.axes[2, -1].figure.canvas.draw_idle()

    @classmethod
    def set_origins(cls, data_path):
        orig = pd.read_csv(data_path, dtype={"evid": str})
        orig["origTime"] = np.array(orig.UTCtime.str[:-1].values, dtype="datetime64[ms]")
        cls.orig = orig
        traces.set_origins(orig)

    @classmethod
    def set_sites(cls, data_path):
        cls.site = pd.read_csv(data_path)

    @classmethod
    def set_stations(cls, st_dic):
        cls.st_dic = pd.DataFrame(st_dic).T

    @classmethod
    def set_tracePath(cls, path):
        cls.tracePath = path

    @classmethod
    def set_workingPickPath(cls, path):
        files = os.listdir(path)
        if len(files) == 0:
            dfs = pd.DataFrame(columns=["plt_num", "pickTime", "Phase", "evid", "confidence", "station"])
        else:
            dfs = []
            for file in files:
                # print(file,files)
                if '_' in file:
                    continue
                # print(os.path.join(path, file),file)
                df = pd.read_csv(os.path.join(path, file), dtype={"event": str, "evid": str})
                df['station'] = file.split('.')[0]
                if 'event' in df.columns:
                    df.rename({'event': 'evid'}, axis=1, inplace=True)
                dfs.append(df)
            dfs = pd.concat(dfs, ignore_index=True)
        dfs['pickTime'] = dfs.pickTime.values.astype('datetime64[ms]')
        dfs["color"] = dfs.Phase.map(picklines)
        cls.pick_df = dfs

    @classmethod
    def set_pickDf(cls, pickDf):
        singleStation_picker.set_save_path(pickDf)
        df = pd.read_csv(pickDf, dtype={"evid": str})
        # print(1,df.columns)
        df['pickTime'] = df.pickTime.values.astype('datetime64[ms]')
        df["color"] = df.Phase.map(picklines)
        cls.save_path = pickDf
        cls.pick_df = df

    @classmethod
    def update_pickDf(cls, df):
        cls.pick_df = df

    @classmethod
    def set_karnelPickNames(cls, Pvalues, Svalues):
        singleStation_picker.set_karnelPickNames(Pvalues, Svalues)

    @classmethod
    def set_time_for_pca(cls, time):
        singleStation_picker.set_time_for_pca(time)

    @classmethod
    def set_stream_length(cls, times):
        singleStation_picker.set_stream_length(times)

    @classmethod
    def set_pca_extent(cls, Min, Max):
        singleStation_picker.set_pca_extent(Min, Max)

    @classmethod
    def set_single_station_path(cls, path):
        if not os.path.exists(path):
            os.makedirs(path)

        cls.df_savePath = path
        singleStation_picker.set_single_station_path(path)

class StatsBuilder:
    def __init__(self, Stats, event):
        self.network = Stats.network
        self.station = Stats.station
        self.location = Stats.location
        self.columnNames = "E,N,Z,delta_times,times".split(",")
        self.starttime = np.datetime64(str(Stats.starttime)[:-1], "ms")
        self.endtime = np.datetime64(str(Stats.endtime)[:-1], "ms")
        self.sampling_rate = Stats.sampling_rate
        self.delta = np.timedelta64(int(Stats.delta * 1000), 'ms')
        self.npts = Stats.npts
        self.instrument = Stats.channel[:2] + "_"
        self.event = event

    def __str__(self):
        """
        Return better readable string representation of Stats object.
        """
        priorized_keys = ['network', 'station', 'location', "instrument", 'columnNames',
                          'starttime', 'endtime', 'sampling_rate', 'delta',
                          'npts', 'instrument', 'event']
        return self._pretty_str(priorized_keys)

    def _pretty_str(self, priorized_keys=[], min_label_length=16):
        """
        Return better readable string representation of AttribDict object.

        :type priorized_keys: list of str, optional
        :param priorized_keys: Keywords of current AttribDict which will be
            shown before all other keywords. Those keywords must exists
            otherwise an exception will be raised. Defaults to empty list.
        :type min_label_length: int, optional
        :param min_label_length: Minimum label length for keywords. Defaults
            to ``16``.
        :return: String representation of current AttribDict object.
        """
        keys = list(priorized_keys)
        # determine longest key name for alignment of all items
        try:
            i = max(max([len(k) for k in keys]), min_label_length)
        except ValueError:
            # no keys
            return ""
        pattern = "%%%ds: %%s" % (i)
        # check if keys exist
        other_keys = [k for k in keys if k not in priorized_keys]
        # priorized keys first + all other keys
        keys = priorized_keys + sorted(other_keys)
        head = [pattern % (k, self.__dict__[k]) for k in keys]
        return "\n".join(head)

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

class traces:

    def __init__(self, path, evid=None, Type=None, fmin=None, fmax=None):
        if not evid:
            eventName = path.split('/')[-1].split('.')[0]
        else:
            eventName = evid

        if fmin:
            self.fmin = fmin
        if fmax:
            self.fmax = fmax

        st = obspy.read(path)

        st.detrend(type="linear")
        st.taper(max_percentage=0.05, type="hann")
        max_start = max([tr.stats.starttime for tr in st])
        min_end = min([tr.stats.endtime for tr in st])
        st.trim(starttime=max_start, endtime=min_end)
        self.stream = st.copy()
        st.filter("bandpass", freqmin=self.fmin, freqmax=self.fmax, corners=4)

        self.stats = StatsBuilder(st[0].stats.copy(), eventName)
        if Type == "vel":
            if self.stats.instrument == "EN_":
                st.integrate()
        elif Type == "disp":
            if self.stats.instrument == "EN_":
                st.integrate()
            st.integrate()


        self.orig_time = self.get_orig_time(eventName)
        dtimes = np.arange(0, len(st[0])) * self.stats.delta.item().total_seconds()

        times = np.arange(self.stats.starttime,
                          self.stats.endtime + self.stats.delta,
                          self.stats.delta,
                          dtype='datetime64[ms]')

        chan_labels = "ENZ"
        arr = np.array([st.select(channel="??" + ch)[0].data for ch in chan_labels])
        a = arr.T
        b = dtimes[:, None]
        c = times.astype("float64")[:, None]
        self.data = np.hstack((a, b, c)).T  # [E,N,Z, delta_times, times]

    @classmethod
    def set_filter_limits(cls, fmin, fmax):
        cls.fmin = fmin
        cls.fmax = fmax

    @classmethod
    def set_origins(cls, orig):
        cls.orig_df = orig.copy()

    def get_orig_time(self, evid):
        return self.orig_df[self.orig_df.evid == evid].iloc[0].origTime


