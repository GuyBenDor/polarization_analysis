import obspy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.optimize import leastsq, minimize, least_squares
from matplotlib.widgets import RadioButtons
import matplotlib as mpl
import matplotlib.dates as mdates

min_snr = 10
min_ps = 0.7
max_inc = 83
min_distance = 0

min_snr = 8
min_ps = 0.7
max_inc = 90
min_distance = 6

anot = True
# anot = False
angle_type = "angle"
angle_type = "epi_direction"

C = ["Ps"]#,"Ps","snr"]#"epi_distance"]#,"timeDelta"]#,"datetime","Ps","epi_distance"]#,"snr","Pp"] #["inclination","Ps","Pp","snr","timeDelta"]

stations = ["GLHS_EN","GLHS_HH"]#,"AMID","DSHN","KDZV","KSH0","LVBS","MTLA","RSPN","KNHM","TVR","HDNS","RMOT","MGDL"]
stations = ["GEM","AMID","LVBS","MTLA","RSP"]
stations = ["LVBS","GEM","MTLA","TVR"]
stations = ["RSPN","DSHN","AMID"]
# stations = ["DSHN"]
# stations = ["AMID"]#,"MTLA"]#,"GEM","NATI"]
# stations = ["GEM","AMID","LVBS","MTLA","RSPN","KSH0","NATI"]
# stations = ("KNHM","TVR","HDNS","RMOT","MGDL")
# stations =["LVBS","GEM"]#,"GEM"]


final_date = 35
start_date = 0
# cmap = plt.cm.jet  # define the colormap
# # extract all colors from the .jet map
# cmaplist = [cmap(i) for i in range(cmap.N)]
# # create the new map
# cmap = mpl.colors.LinearSegmentedColormap.from_list(
#     'Custom cmap', cmaplist, cmap.N)
# # define the bins and normalize
# cbounds = np.arange(start_date, final_date+1, 1)
# norm = mpl.colors.BoundaryNorm(cbounds, cmap.N)


def make_fig(data,station,angle_type,fit,c,fig, zero_cross=None,cmap=None, norm=None):
    global C
    ax = fig.add_subplot()
    # print(data.columns)
    divider = make_axes_locatable(ax)
    if cmap and norm:
        pos = ax.scatter(
            data[angle_type], data["offset"],
            s=30, c=data[c], zorder=2, edgecolor="k", lw=0.5, cmap=cmap, norm=norm
        )

    else:
        # print(data[c])
        pos = ax.scatter(data[angle_type], data["offset"], s=30, c=data[c], zorder=2, edgecolor="k", lw=0.5)

    ax_histy = divider.append_axes("right", 0.6, pad=0.1, sharey=ax)
    ax_histy.yaxis.set_tick_params(labelleft=False)
    binwidth = 10
    bins = np.arange(-185, 185 + binwidth, binwidth)
    ax_histy.hist(data["offset"], bins=bins, orientation='horizontal')
    cb = fig.colorbar(mappable=pos, ax=ax, label=c)
    # if c=="pickTime":
    #     cb.set_ticks([pos.colorbar.vmin + t * (pos.colorbar.vmax - pos.colorbar.vmin) for t in cb.ax.get_yticks()])
    #     cb.set_ticklabels([mdates.datetime.datetime.fromtimestamp(
    #         (pos.colorbar.vmin + t * (pos.colorbar.vmax - pos.colorbar.vmin)) / 1000000000).strftime('%c') for t in
    #                          cb.ax.get_yticks()])

    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(0, 360)
    ax.set_title(station)
    ax.set_ylim(-180, 180)

    # ax.plot(np.degrees(fine_t), data_l1, label="l1 fit (mav)")

    # ax.plot(np.degrees(fine_t), data_l2-output2.x[2], label="l2 fit (lsq)", zorder=1, c="green")
    ax.axhline(y=0, c='k', linestyle="--", zorder=1, label="zero offset")
    if isinstance(zero_cross,np.ndarray):
        ax.plot(np.degrees(fit[0]), fit[1], label="l2 fit (lsq)", zorder=1, c="red")
        ax.scatter(zero_cross, np.zeros(zero_cross.shape), marker="*", edgecolor="k", c="yellow", lw=0.5, zorder=3, s=130,
                   label="zero cross")
    ax.legend(prop={'size': 6})
    ax.set_xlabel("epicenter azimuth [" + "$^\circ]$")

    ax.set_ylabel("offset [" + "$^\circ]$")

    return ax,cb,pos,ax_histy




for i,station in enumerate(stations):
    data = pd.read_csv(f"pWaveParam/{station}.csv")
    # data = pd.read_csv(f"GLHS_rotation/{station}.csv")
    data = data[
        (data.snr>=min_snr) &
        (data.Ps>=min_ps) &
        (data.inclination<=max_inc) &
        (data.epi_distance >= min_distance)
    ]
    data["snr"] = np.where(data.snr<50,data.snr,50)
    data["pickTime"] = pd.to_datetime(data["pickTime"], format='%Y-%m-%d %H:%M:%S.%f')
    data["timeDelta"] = (data["pickTime"] - data["pickTime"].min()).dt.days
    # data["datetime"] = data.plt_num-data.plt_num.min()
    # data["timeDelta"] = np.where(data.timeDelta<final_date,data.timeDelta,final_date)
    # print(data.columns)
    # data = data[data["timeDelta"]<final_date]
    # if start_date:
    #     data = data[data["timeDelta"] >= start_date]
    # print(data[np.abs(data.offset)>120].event.values)

    ######
    t = np.radians(data[angle_type].to_numpy())
    # t = np.radians(data["angle"].to_numpy())
    y = data["offset"].to_numpy()

    # print(data[data.offset>=40].event.values)

    guess_mean = y.mean()
    guess_phase = 0
    guess_amp = 1

    x0 = np.array([guess_amp, guess_phase, guess_mean])
    output1, freq1, objective1 = None, None, 10e10
    output2, freq2, objective2 = None, None, 10e10
    bounds = [[-180, 180], [0, 2 * np.pi], [-180, 180]]
    for freq in [1, 2]:
        optimize_func1 = lambda x: np.absolute(x[0] * np.sin(t * freq + x[1]) + x[2] - y).sum()
        temp = minimize(optimize_func1, x0, bounds=bounds)
        # print(temp.fun)
        if temp.fun < objective1:
            output1, freq1 = temp, freq
            objective1 = temp.fun

        optimize_func2 = lambda x: ((x[0] * np.sin(t * freq + x[1]) + x[2] - y) ** 2).sum()
        temp = minimize(optimize_func2, x0, bounds=bounds)
        # print(temp.fun)
        if temp.fun < objective2:
            output2, freq2 = temp, freq
            objective2 = temp.fun

    fine_t = np.arange(-np.pi, 3 * np.pi, 0.1)
    data_l1 = output1.x[0] * np.sin(fine_t * freq1 + output1.x[1]) + output1.x[2]
    data_l2 = output2.x[0] * np.sin(fine_t * freq2 + output2.x[1]) + output2.x[2]

    res = (np.degrees(np.arcsin(1/output2.x[0])-output2.x[1])/freq2)

    deg = np.arange(0,360,90) if freq2==2 else np.arange(0,360,180)
    res = res%90 if freq2==2 else res%180

    zero_cross = deg+res
    # print(output2.x[2])
    # print(zero_cross)
    if station=="RSPN":
        zero_cross=None
    # print(data.columns)
    if anot:
        for c in C:
            fig = plt.figure()
            ax, cb, sc, ax_histy= make_fig(data, station, angle_type,[fine_t, data_l2], c, fig, zero_cross=zero_cross)
            annot = ax_histy.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",fontsize=8,xycoords=ax.transData,
                                bbox=dict(boxstyle="round", fc="w"),
                                arrowprops=dict(arrowstyle="->"))
            annot.set_visible(False)

            def update_annot(ind):
                pos = sc.get_offsets()[ind["ind"][0]]
                row = data.iloc[ind["ind"]]
                annot.xy = pos
                # print(row.evid)
                text = f"evid: {row.evid.values[0]}\nPs: {round(row.Ps.values[0],2)}" \
                       f"\nsnr: {round(row.snr.values[0],2)}\ninc: {round(row.inclination.values[0],2)}" \
                       f"\noffset: {round(row.offset.values[0],2)}\nepi_dist: {round(row.epi_distance.values[0],2)}"
                # text = "{}, {}".format(" ".join(list(map(str, ind["ind"]))),
                #                        " ".join([names[n] for n in ind["ind"]]))
                annot.set_text(text)
                # annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
                annot.get_bbox_patch().set_alpha(0.8)


            def hover(event):
                vis = annot.get_visible()
                if event.inaxes == ax:
                    cont, ind = sc.contains(event)
                    if cont:
                        update_annot(ind)
                        annot.set_visible(True)
                        fig.canvas.draw_idle()
                    else:
                        if vis:
                            annot.set_visible(False)
                            fig.canvas.draw_idle()


            fig.canvas.mpl_connect("motion_notify_event", hover)

            plt.show()

    else:
        for c in C:
            fig = plt.figure()
            ax, cb, sc,ax_histy = make_fig(data, station, angle_type,[fine_t, data_l2], c, fig, zero_cross=zero_cross)#,cmap=cmap,norm=norm)

# props = dict(boxstyle='round', facecolor='white', alpha=0.5)
# fig, axes = plt.subplots(nrows=3, ncols=2,figsize=(10,9))
# for i,station in enumerate(stations):
#     data = pd.read_csv(f"p_wave_data/{station}.csv")
#     data = data[
#         (data.snr>=min_snr) &
#         (data.Ps>=min_ps) &
#         (data.inclination<=max_inc)
#     ]
#     data["snr"] = np.where(data.snr<50,data.snr,50)
#     data["pickTime"] = pd.to_datetime(data["pickTime"], format='%Y-%m-%d %H:%M:%S.%f')
#     data["timeDelta"] = (data["pickTime"] - data["pickTime"].min()).dt.days
#     data["datetime"] = data.plt_num-data.plt_num.min()
#     # data = data[data["datetime"]>50]
#
#     ######
#     t = np.radians(data["epi_direction"].to_numpy())
#     y = data["offset"].to_numpy()
#
#     guess_mean = y.mean()
#     guess_phase = 0
#     guess_amp = 1
#
#     x0 = np.array([guess_amp, guess_phase, guess_mean])
#     output1, freq1, objective1 = None, None, 10e10
#     output2, freq2, objective2 = None, None, 10e10
#     bounds = [[-180, 180], [0, 2 * np.pi], [-180, 180]]
#     for freq in [1, 2]:
#         optimize_func1 = lambda x: np.absolute(x[0] * np.sin(t * freq + x[1]) + x[2] - y).sum()
#         temp = minimize(optimize_func1, x0, bounds=bounds)
#         # print(temp.fun)
#         if temp.fun < objective1:
#             output1, freq1 = temp, freq
#             objective1 = temp.fun
#
#         optimize_func2 = lambda x: ((x[0] * np.sin(t * freq + x[1]) + x[2] - y) ** 2).sum()
#         temp = minimize(optimize_func2, x0, bounds=bounds)
#         # print(temp.fun)
#         if temp.fun < objective2:
#             output2, freq2 = temp, freq
#             objective2 = temp.fun
#
#     fine_t = np.arange(-np.pi, 3 * np.pi, 0.1)
#     data_l1 = output1.x[0] * np.sin(fine_t * freq1 + output1.x[1]) + output1.x[2]
#     data_l2 = output2.x[0] * np.sin(fine_t * freq2 + output2.x[1]) + output2.x[2]
#
#     res = (np.degrees(np.arcsin(1/output2.x[0])-output2.x[1])/freq2)%180
#
#     zero_cross = [res,res+180]
#     # print(output2.x[2])
#     # print(zero_cross)
#     if station=="RSPN":
#         zero_cross=None
#
#     axes.flat[i], cb = make_subplot_fig(data, station, [fine_t, data_l2], C[0], axes.flat[i], zero_cross=zero_cross)
#
#     axes.flat[i].vlines(x=[90,180,270],ymin=-180,ymax=180,color="grey",alpha=0.2,linestyle="--")
#
#     axes.flat[i].hlines(y=[-120, -60, 60, 120], xmin=0, xmax=360, color="grey", alpha=0.2, linestyle="--")
#
#     # ax.legend(prop={'size': 6})
#     if i in [3,4]:
#         axes.flat[i].set_xlabel("epicenter azimuth [" + "$^\circ]$")
#         axes.flat[i].set_xticks(np.arange(0,361,90))
#     else:
#         axes.flat[i].axes.xaxis.set_visible(False)
#
#
#     if i in [0,2,4]:
#         axes.flat[i].set_ylabel("offset [" + "$^\circ]$")
#         axes.flat[i].set_yticks(np.arange(-180, 181, 60))
#     else:
#         axes.flat[i].axes.yaxis.set_visible(False)
#     axes.flat[i].text(0.05, 0.95, station, transform=axes.flat[i].transAxes, fontsize=8,
#         verticalalignment='top', bbox=props)
#     # ax.set_ylabel("offset [" + "$^\circ]$")
# fig.delaxes(axes.flat[5])
# fig.subplots_adjust(wspace=0.0,hspace=0.2,right=0.88,left=0.292)
# cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.7])
# cbar = fig.colorbar(cb, cax=cbar_ax)
# cbar.set_ticks(np.arange(0.7,1.1,0.1))
# cbar.set_ticklabels(np.arange(0.7,1.1,0.1))
# cbar.set_label('Rectilinearity',loc="center")#,rotation=360)
# handles, labels = axes.flat[0].get_legend_handles_labels()
# fig.legend(handles, labels, loc=[0.6,0.2])

plt.show()




