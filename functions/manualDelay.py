from matplotlib import pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.widgets import Button, CheckButtons, RadioButtons,MultiCursor,TextBox,Slider, RangeSlider
from matplotlib.gridspec import GridSpec
import threading
from functions.baseFunctions import find_nearest, mat_rotation, tracesMan, heatmap, annotate_heatmap


cursor_colors = ["tab:green", "tab:blue", "tab:red"]
DEBOUNCE_DUR = 2

class manuel_delay:
    def __init__(self, st, rotation_type, **kwargs):

        # self.tw_lim = []
        self.obj = st
        self.rotation_type = rotation_type
        self.__dict__.update(kwargs)

        self.set_save_df()
        self.catch_event()
        self.get_inx_of_important_times()
        self.press = False
        self.move = False
        self.active_span = False
        self.saveText = False

        self.x_start = self.obj.st.data[-2, self.pickinx]
        if not np.isnan(self.eventPicks.man_delay):
            self.delay_time = round(self.eventPicks.man_delay, 3)
            self.delay = int(self.eventPicks.man_delay * self.obj.st.stats.sampling_rate)
        else:
            self.delay_time = self.delay = 0
        # print(self.delay,)

        if not np.isnan(self.eventPicks.error):
            self.error = self.eventPicks.error
        else:
            self.error = np.nan

        # self.delay_time = self.delay = 0
        self.rotAngle = self.eventPicks[f'angle_{self.rotation_type}']
        # print(self.eventPicks)
        self.set_figure()
        self.plot_traces_in_figure()
        self.plot_motion_in_figure()
        self.set_heat_map()

        self.plot_updated_delay(self.delay_time)

        if not np.isnan(self.error):
            self.spansE = [ax.axvspan(self.x_start - self.error,
                                      self.x_start + self.error, alpha=0.1, color='red')
                           for ax in
                           self.axes[:-2]]


        self.set_buttons()
        self.set_interaction()

    def set_interaction(self):
        # self.picking = self.picking_buttons.on_clicked(self.set_pick_status)
        self.filter = self.filt_rangeSlider.on_changed(self.update_filter)
        self.rotation = self.rotation_slider.on_changed(self.update_rotation)
        self.tw = self.pm_tw_rangeSlider.on_changed(self.update_tw)
        self.dl = self.delay_slider.on_changed(self.update_delay)
        self.rst = self.reset_button.on_clicked(self.reset)
        self.delete = self.delete_button.on_clicked(self.delete_error)
        self.save = self.save_picking_button.on_clicked(self.save_data)
        self.multi = MultiCursor(
            self.fig.canvas, (self.axes[0], self.axes[1], self.axes[2]), color=cursor_colors[0], lw=1
        )

        self.c4 = self.fig.canvas.mpl_connect('key_press_event', self.onpresskey)
        # self.save = self.save_picking_button.on_clicked(self.save_data_frame)
        # self.connection_id = self.fig.canvas.mpl_connect('button_press_event', self.onclick)
        plt.show()

    def set_buttons(self):
        save_bot_pos = [0.94, 0.25 + 0.035*4, 0.035, 0.035]
        pm_tw_range_pos = [0.4, 0.1, 0.5, 0.02]
        reset_pos = [0.8, 0.025, 0.1, 0.04]
        delay_range_pos = [0.4, 0.08, 0.5, 0.02]
        filt_range_pos = [0.4, 0.89, 0.5, 0.02]
        rotation_range_pos = [0.92, 0.15, 0.02, 0.7]
        delete_err_pos = [0.69, 0.025, 0.1, 0.04]
        axcolor = 'lightgoldenrodyellow'

        self.saveBut = plt.axes(save_bot_pos)
        self.filt_ax = plt.axes(filt_range_pos)
        self.rot_ax = plt.axes(rotation_range_pos)
        self.pm_tw_ax = plt.axes(pm_tw_range_pos)
        self.delay_ax = plt.axes(delay_range_pos)
        self.reset_ax = plt.axes(reset_pos)
        self.delete_ax = plt.axes(delete_err_pos)

        self.filt_rangeSlider = RangeSlider(
            self.filt_ax, "filter", 0.2, 50, valinit=self.filterValues, orientation="horizontal", valstep=0.1
        )

        self.rotation_slider = Slider(
            ax=self.rot_ax, label="rotation", valmin=0, valmax=359,
            valinit=self.eventPicks[f'angle_{self.rotation_type}'], orientation="vertical", valstep=1
        )

        self.pm_tw_rangeSlider = RangeSlider(
            self.pm_tw_ax, "pm time-window", -5, 5,
            valinit=[-self.stream_length[0], self.stream_length[1]],
            orientation="horizontal", valstep=1/self.obj.st.stats.sampling_rate
        )
        self.delay_slider = Slider(
            ax=self.delay_ax, label="delay time", valmin=-0.5, valmax=0.5, valinit=self.delay_time, orientation="horizontal",
            valstep=1./self.obj.st.stats.sampling_rate
        )

        self.save_picking_button = Button(self.saveBut, 'Save', hovercolor='0.975')

        self.reset_button = Button(self.reset_ax, 'Reset', hovercolor='0.975')

        self.delete_button = Button(self.delete_ax, 'delete error', hovercolor='0.975')

    def delete_error(self, event):
        if not self.active_span and hasattr(self, "spansE"):
            for span, ax in zip(self.spansE, self.axes[:-2]):
                span.remove()
                ax.figure.canvas.draw_idle()
            delattr(self, "spansE")
            self.error = np.nan

    def set_figure(self):
        self.fig = plt.figure(figsize=(15, 8))

        gs = GridSpec(7, 18, figure=self.fig)

        self.axes = [self.fig.add_subplot(gs[step:step + 2, 7:]) for step in range(0, 6, 2)]
        if hasattr(self, 'fig_3d') and not self.fig_3d:
            self.axes += [self.fig.add_subplot(gs[:5, :5])]
        else:
            self.axes += [self.fig.add_subplot(gs[:5, :6], projection='3d')]
        self.axes += [self.fig.add_subplot(gs[5:, :3])]

        self.textBox = self.axes[-2].annotate("",
                              xy=(0, 0), xytext=(0.1, 1), fontsize=10,
                              xycoords='axes fraction', textcoords='axes fraction',
                              bbox=dict(boxstyle="round",
                                          ec=(1., 0.5, 0.5)),
                              horizontalalignment='center', verticalalignment='center')
        self.update_textBox()

        self.saveTextBox = self.axes[2].annotate("data was saved",
                                              xy=(0, 0), xytext=(0.5, 1.5), fontsize=30, rotation=30,
                                              xycoords='axes fraction', textcoords='axes fraction',
                                              bbox=dict(boxstyle="round",facecolor='white', alpha=0.8),
                                              horizontalalignment='center', verticalalignment='center')

        self.saveTextBox.set_visible(self.saveText)


        plt.subplots_adjust(hspace=0)
        plt.suptitle(f'{self.obj.st.stats.station} - {self.obj.st.stats.event}')

    def set_heat_map_cbar(self):
        # Create colorbar
        self.cbar = self.axes[-1].figure.colorbar(self.im, ax=self.axes[-1])
        self.cbar.ax.set_ylabel("", rotation=-90, va="bottom")

    def get_inx_of_important_times(self):
        vals = [
            np.datetime64(self.eventPicks.pickTime),
            np.datetime64(self.obj.st.orig_time),
            np.datetime64(self.eventPicks.pickTime) - np.timedelta64(int(self.stream_length[0]*1000), "ms"),
            np.datetime64(self.eventPicks.pickTime) + np.timedelta64(int(self.stream_length[1]*1000), "ms")
        ]
        self.limlist = vals.copy()[2:]
        vals = [find_nearest(self.obj.st.data[-1].astype("datetime64[ms]"), lim) for lim in vals]
        self.pickinx = vals[0]
        self.originx = vals[1]
        # if not all(np.isnan([self.eventPicks.tw_ind_min, self.eventPicks.tw_ind_max])):
        #     self.limlistinx = np.array([self.eventPicks.tw_ind_min, self.eventPicks.tw_ind_max],dtype=int)
        # else:
        #     self.limlistinx = vals[2:]
        self.limlistinx = vals[2:]

    def plot_traces_in_figure(self):

        self.update_data()
        self.ls = []
        self.spans = []
        # td = timedelta(milliseconds=1000 / self.obj.st.stats.sampling_rate)
        # self.datetimeRange = drange(
        #     datetime.strptime(str(self.obj.st.stats.starttime), "%Y-%m-%dT%H:%M:%S.%f"),
        #     datetime.strptime(str(self.obj.st.stats.endtime), "%Y-%m-%dT%H:%M:%S.%f")+td,
        #     td
        # )
        self.trace_names = ['slow', 'fast', 'vertical']
        for i in range(3):
            # l, = self.axes[i].plot(self.datetimeRange, self.data[i], "k", lw=0.5)
            l, = self.axes[i].plot(self.data[-2], self.data[i], "k", lw=0.5)
            # self.axes[i].axvline(x=self.originx, color="lime", ls="--")
            self.ls.append(l)
            self.axes[i].annotate(self.trace_names[i],
                                  xy=(0, 0), xytext=(0.1, 0.9), fontsize=10,
                                  xycoords='axes fraction', textcoords='axes fraction',
                                  bbox=dict(facecolor='white', alpha=0.8),
                                  horizontalalignment='right', verticalalignment='top')
            span = self.axes[i].axvspan(self.data[-2,self.limlistinx[0]],
                                 self.data[-2,self.limlistinx[1]],
                                 alpha=0.1, color='green')
            self.spans.append(span)

            if i != 2:
                self.axes[i].set_xticks([])
        # self.picklines = {}

        self.plot_lineCollection()

        # print(self.eventPicks)
        self.axes[2].get_shared_x_axes().join(self.axes[0], self.axes[1], self.axes[2])
        self.axes[2].get_shared_y_axes().join(self.axes[0], self.axes[1], self.axes[2])


        self.axes[2].set_xlim(
            self.data[-2,self.limlistinx[0]]-1,
            self.data[-2,self.limlistinx[1]]+1
        )
        ilim = np.where(
            (self.data[-1].astype("datetime64[ms]") > self.limlist[0]) &
            (self.data[-1].astype("datetime64[ms]") < self.limlist[1])
        )[0]
        if len(ilim) == 0:
            return
        y_max = np.max(np.abs(self.data[:3, ilim])) * 1.2
        self.lsylim = [-y_max, y_max]
        self.axes[2].set_ylim(self.lsylim[0], self.lsylim[1])

    def plot_lineCollection(self):
        # print(self.eventPicks.status.values)
        # colors = [cursor_colors[val] for val in self.eventPicks.status.values]
        picklines = [self.data[-2, self.originx], self.data[-2, self.pickinx]]

        seg = np.zeros((len(picklines),2, 2))
        seg[:, :, 0] = np.c_[picklines,picklines]
        seg[:, :, 1] = [-1, 1]

        for i in range(3):
            for col in self.axes[i].collections:
                col.remove()
            lc = LineCollection(
                seg, colors=["lime", cursor_colors[2]], transform=self.axes[i].get_xaxis_transform()#, label=row.Phase
            )
            self.axes[i].add_collection(lc)
            self.axes[i].figure.canvas.draw()

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

    def update_pm_data(self):
        self.update_data()
        if self.delay==0:
            arr = self.data[1:]
            arr0 = self.data[0]
        elif self.delay>0:
            arr = self.data[1:, :-self.delay]
            arr0 = self.data[0, self.delay:]
        else:
            arr = self.data[1:, -self.delay:]
            arr0 = self.data[0, :self.delay]
        data = np.concatenate((arr0[:, np.newaxis].T, arr), 0)
        dtimes, times = data[3:, :]
        dtimes = dtimes[:, None]
        times = times[:, None]
        pm_arr = mat_rotation(-self.rotAngle, data[:3])
        self.pm_arr = np.hstack((pm_arr.T, dtimes, times)).T
        self.figlim = 1.1 * np.sqrt((self.pm_arr[:3, self.limlistinx[0]:self.limlistinx[1]] ** 2).sum(axis=0)).max()

    def set_save_df(self):
        cols = ["man_delay", "corr_val", "fmin", "fmax", "tw_ind_min", "tw_ind_max", "man_rotation", "error"]
        for col in cols:
            if not col in tracesMan.df.columns:
                tracesMan.df[col] = np.nan
        self.save_df = tracesMan.df.copy()

    def catch_event(self):
        eventPicks = self.save_df.copy()[
            (self.save_df.evid == self.obj.st.stats.event) &
            (self.save_df.station == self.obj.st.stats.station)
        ]
        self.save_df.drop(index=eventPicks.index, inplace=True)
        self.save_df.reset_index(inplace=True, drop=True)
        self.eventPicks = eventPicks.iloc[0]

    def plot_motion_in_figure(self):
        self.update_pm_data()
        self.particle_motion()

    def particle_motion(self):
        motion = self.pm_arr[:, self.limlistinx[0]:self.limlistinx[1]]
        self.figlim = 1.1 * np.sqrt((self.pm_arr[:3, self.limlistinx[0]:self.limlistinx[1]] ** 2).sum(axis=0)).max()
        zeros = np.zeros([2, 3])
        zeros[0, 0] = self.figlim * np.sin(np.radians([self.rotAngle]))
        zeros[0, 1] = self.figlim * np.cos(np.radians([self.rotAngle]))
        if not hasattr(self, "fig_3d") or self.fig_3d:
            self.line3d, = self.axes[3].plot(motion[0], motion[1], motion[2], "k", lw=0.5)
            self.axes[3].set_zlim(-self.figlim, self.figlim)
            self.rotation_line, = self.axes[3].plot(zeros.T[0], zeros.T[1], zeros.T[2], "cyan", alpha=0.7)
        else:
            self.line3d, = self.axes[3].plot(motion[0], motion[1], "k", lw=0.5)
            self.rotation_line, = self.axes[3].plot(zeros.T[0], zeros.T[1], "cyan", alpha=0.7)
            self.axes[3].set_aspect('equal')
        self.axes[3].set_xlim(-self.figlim, self.figlim)
        self.axes[3].set_ylim(-self.figlim, self.figlim)

    def update_pm(self):
        self.line3d.set_data_3d(
            self.pm_arr[0, self.limlistinx[0]:self.limlistinx[1]],
            self.pm_arr[1, self.limlistinx[0]:self.limlistinx[1]],
            self.pm_arr[2, self.limlistinx[0]:self.limlistinx[1]]
        )
        if not hasattr(self, "fig_3d") or self.fig_3d:
            self.axes[3].set_zlim(-self.figlim, self.figlim)
        self.axes[3].set_xlim(-self.figlim, self.figlim)
        self.axes[3].set_ylim(-self.figlim, self.figlim)
        self.set_heat_map()

    def update_delay(self, val):
        # self.eventPicks["man_delay"] = val
        self.delay_time = round(val, 3)
        self.delay = int(val*self.obj.st.stats.sampling_rate)
        # self.ls[0].set_ydata(self.data[0,self.limlistinx[0]+val:self.self.limlistinx[1]])
        self.plot_updated_delay(val)

    def plot_updated_delay(self, val):
        self.ls[0].set_xdata(self.data[-2]-val)
        self.update_pm_data()
        self.update_pm()
        # pass

    def update_filter(self, val):
        self.filterValues = val
        print(f"bandpass {val[0]}-{val[1]} Hz filter applied")
        self.update_data()
        for i in range(3):
            self.ls[i].set_ydata(self.data[i])

        self.update_pm_data()
        if hasattr(self, "line3d"):
            self.update_pm()

        # self.plot_motion_in_figure()
        # self.fig.canvas.draw_idle()

    def update_rotation(self, val):
        self.rotAngle = val
        print(f"rotation of {val} applied")
        self.update_data()
        for i in range(3):
            self.ls[i].set_ydata(self.data[i])

        zeros = np.zeros([2, 3])
        zeros[0, 0] = self.figlim * np.sin(np.radians([self.rotAngle]))
        zeros[0, 1] = self.figlim * np.cos(np.radians([self.rotAngle]))

        self.rotation_line.set_data_3d(zeros.T[0], zeros.T[1], zeros.T[2])

        self.fig.canvas.draw_idle()

    def set_heat_map(self):
        self.axes[-1].cla()
        self.corrcoef = np.round_(np.corrcoef(self.pm_arr[:3, self.limlistinx[0]:self.limlistinx[1]]), 2)
        # self.recti =
        self.im = heatmap(self.corrcoef, list(self.trace_names), list(self.trace_names), ax=self.axes[-1],
                                   cmap="PuOr", vmin=-1, vmax=1,
                                   cbarlabel="correlation coeff.")
        self.heat_map_text = annotate_heatmap(self.im)

    def reset(self, event):
        self.filt_rangeSlider.set_val(self.filt_init)
        self.rotation_slider.reset()
        self.pm_tw_rangeSlider.set_val(self.stream_length_init)
        self.delay_slider.reset()

    def update_tw(self, val):
        vals = np.array(val * 1000, dtype='timedelta64[ms]') + np.datetime64(self.eventPicks.pickTime)
        self.limlistinx = [find_nearest(self.data[-1].astype("datetime64[ms]"), t) for t in vals]
        # self.limlistinx = [find_nearest(self.pm_arr[-1].astype("datetime64[ms]"), t) for t in vals]
        for span in self.spans:
            span.set_xy([[self.data[-2, self.limlistinx[0]], 0],
                         [self.data[-2, self.limlistinx[0]], 1],
                         [self.data[-2, self.limlistinx[1]], 1],
                         [self.data[-2, self.limlistinx[1]], 0],
                         [self.data[-2, self.limlistinx[0]], 0]])
        self.update_pm()

    def save_data(self, event):
        tracesMan.df.loc[
            tracesMan.df[
                (tracesMan.df.evid == self.eventPicks.evid) &
                (tracesMan.df.station == self.eventPicks.station)
                ].index,
            ["man_delay", "corr_val", "fmin", "fmax", "tw_ind_min", "tw_ind_max", "man_rotation", "error"]
        ] = [
            self.delay_time, self.corrcoef[0, 1], self.filterValues[0],
            self.filterValues[1], self.limlistinx[0], self.limlistinx[1], self.rotAngle, self.error
        ]
        if not hasattr(self.obj, "save_path"):
            print("Need to set save_path")
            return

        print("The following values are saved:")
        print(tracesMan.df.loc[
            tracesMan.df[
                (tracesMan.df.evid == self.eventPicks.evid) &
                (tracesMan.df.station == self.eventPicks.station)
                ].index,
            ['man_delay', 'corr_val', 'fmin', 'fmax', 'tw_ind_min', 'tw_ind_max','man_rotation', 'error']
        ].iloc[0])
        tracesMan.df.to_csv(self.obj.save_path, index=False)
        self.add_notify_dataSave()
        threading.Timer(DEBOUNCE_DUR, self.remove_notify_dataSave).start()

    def onpresskey(self, event):
        if event.key == 't':

            if self.active_span:
                # print('Key pressed. Deactivated.')
                self.active_span = False
                self.fig.canvas.mpl_disconnect(self.c1)
                self.fig.canvas.mpl_disconnect(self.c2)
                self.fig.canvas.mpl_disconnect(self.c3)
            else:
                # print('Key pressed. Activated.')
                self.active_span = True
                if hasattr(self, "spansE"):
                    for span in self.spansE:
                        span.remove()
                    delattr(self, "spansE")
                self.c1 = self.fig.canvas.mpl_connect('button_press_event', self.onpress)
                self.c2 = self.fig.canvas.mpl_connect('button_release_event', self.onrelease)
                self.c3 = self.fig.canvas.mpl_connect('motion_notify_event', self.onmotion)
                # print(len(self.axes))

                val = self.error if not np.isnan(self.error) else 1 / self.obj.st.stats.sampling_rate

                self.spansE = [ax.axvspan(self.x_start - val,
                                          self.x_start + val, alpha=0.1, color='red')
                               for ax in
                               self.axes[:-2]]
            self.update_textBox()

        elif event.key == "ctrl+c":
            self.save_data(event)

    def onpress(self, event):
        if event.button == 3:
            self.press = True

    def onmotion(self, event):
        'on motion we will move the rect if the mouse is over us'
        if not any([0 if event.inaxes != ax else 1 for ax in self.axes[:-2]]): return
        if not self.press: return
        self.move = True
        dx = abs(event.xdata - self.x_start)
        start = self.x_start - dx
        end = self.x_start + dx
        y = self.spansE[0].get_xy()[:, 1]
        x = [start, start, end, end, start]
        d = np.array([x, y]).T
        for span, ax in zip(self.spansE, self.axes[:-2]):
            span.set_xy(d)
            ax.figure.canvas.draw_idle()

    def onrelease(self, event):
        if self.press and not self.move:
            # self.delete_error()
            pass
        elif self.press and self.move:
            self.move = False
            self.error = abs(self.x_start - event.xdata)
            # print("finished", self.error)
        self.press = False

    def update_textBox(self):
        if self.active_span:
            self.textBox.set_text("error mode activated")
            self.textBox.set_backgroundcolor((0.8, 1., 0.8))
        else:
            self.textBox.set_text("error mode deactivated")
            self.textBox.set_backgroundcolor((1., 0.8, 0.8))
        self.axes[-2].figure.canvas.draw_idle()

    def add_notify_dataSave(self):
        self.saveText = True
        self.saveTextBox.set_visible(self.saveText)
        self.axes[2].figure.canvas.draw_idle()

    def remove_notify_dataSave(self):
        self.saveText = False
        self.saveTextBox.set_visible(self.saveText)
        self.axes[2].figure.canvas.draw_idle()

    @classmethod
    def set_filterValues(cls, minMax):
        cls.filt_init = cls.filterValues = minMax

    @classmethod
    def set_stream_length(cls, times):
        cls.stream_length_init = cls.stream_length = times

class Figure:

    def __init__(self, path, rotation='2d',Type=None, **kwargs):
        self.st = tracesMan(path, Type=Type)
        # self.pick_df = self.get_picks(save_path)
        self.fc = manuel_delay(self, rotation, **kwargs)

    @classmethod
    def set_data(cls,path):
        tracesMan.set_data(path)

    @classmethod
    def set_filter_values(cls,vals):
        manuel_delay.set_filterValues(vals)
        tracesMan.set_filter_limits(vals[0], vals[1])

    @classmethod
    def set_stream_length(cls,times):
        manuel_delay.set_stream_length(times)

    @classmethod
    def set_save_path(cls, save_path):
        cls.save_path = save_path

