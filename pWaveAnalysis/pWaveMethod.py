import pandas as pd
import matplotlib.pyplot as plt
from functions.pWave import P_wave_analysis, plot_3d_motion, plot_2d_motions
import os

cm = plt.cm.get_cmap('Greys')

# test = [229.8,219.9,226.9,185.2]
stations = [i.split(".")[0] for i in os.listdir("../pickingEvents/pickFiles") if "_" not in i]


stations = ["GEM","AMID","LVBS","MTLA","RSPN","KSH0","NATI"]#["LVBS","GEM","MTLA","TVR"]#,"HDNS"]#,"GLHS","MTLA"]#,"GEM"]
stations = ["DSHN"]
Type=None
EVENT = "124522"

plot = True
# plot = False

use_hh = True
# use_hh = False


special_cases = {
    "GEM": {
        "ch": "HH",
        "filt":[2,8]
    },
    "GLHS": {
        "ch": "HH",
        "filt":[2,8]
    }
}




P_wave_analysis.set_time_before_pick(2)
P_wave_analysis.set_half_cycle_num(2)
P_wave_analysis.set_catalog("../data/relocated3.csv")
P_wave_analysis.set_sites("../data/sites.csv")

P_wave_analysis.set_time_for_noise_level(3)
P_wave_analysis.set_noise_multiple(4)



# corr_rotated.set_origins("/Users/guy.bendor/PycharmProjects/ShearWave_splitting/data/main_data_files/relocated.csv")
# corr_rotated.set_sites("/Users/guy.bendor/PycharmProjects/ShearWave_splitting/data/main_data_files/sites.csv")

for station in stations:
    print(station)

    channels = "EN"
    filt = [1, 8]

    if station in special_cases.keys():
        if use_hh:
            channels = special_cases[station]['ch']
            filt = special_cases[station]['filt']


    station = station.split(".")[0]
    pick_path = f"../pickingEvents/pickFiles/{station}.csv"
    P_wave_analysis.set_picks(pick_path,"P")#,area=area)

    # pick_df = pd.read_csv(pick_path,dtype={"event":str,"evid":str})
    # if "event" in pick_df.columns:
    #     pick_df.rename(columns={"event": "evid"},inplace=True)


    # pick_df = pick_df[pick_df["Phase"]=="P"].evid.values

    trace_path = f"/Users/guy.bendor/Traces/StreamForPca/{station}/{channels}/"
    P_wave_analysis.set_filter([filt[0], filt[1]])

    obj = P_wave_analysis(station, trace_path,Type=Type)
    if plot:
        if "EVENT" in globals():
            obj.streams[EVENT].dic = {
                "components": obj.dic[EVENT],
                "angle": obj.data[obj.data.evid == EVENT].angle.values[0],
                "inclination": obj.data[obj.data.evid == EVENT].inclination.values[0]
            }
            # obj.streams[EVENT].Vdic = {"components": obj.Vdic[EVENT]}
            plot_3d_motion(obj.streams[EVENT], obj.data[obj.data.evid == EVENT])
            plot_2d_motions(obj.streams[EVENT], obj.data[obj.data.evid == EVENT])
            print("epi_direction: ", obj.data[obj.data.evid == EVENT].epi_direction.values)
            print("angle: ", obj.data[obj.data.evid == EVENT].angle.values)
            print("offset: ", obj.data[obj.data.evid == EVENT].offset.values)
            print("snr: ", obj.data[obj.data.evid == EVENT].snr.values)
            print("Ps: ", obj.data[obj.data.evid == EVENT].Ps.values)
            print("Pp: ", obj.data[obj.data.evid == EVENT].Pp.values)
            print("inclination: ", obj.data[obj.data.evid == EVENT].inclination.values)
            plt.show()
        else:
            for key in obj.streams.keys():
                print(key)
                # print(obj.streams[key].stats)
                #print("evid: ", obj.data[obj.data.evid == key].evid.values[0])
                # print(obj.__dict__)
                obj.streams[key].dic = {
                    "components": obj.dic[key],
                    "angle": obj.data[obj.data.evid == key].angle.values[0],
                    "inclination": obj.data[obj.data.evid == key].inclination.values[0]
                }
                plot_3d_motion(obj.streams[key], obj.data[obj.data.evid == key])
                plot_2d_motions(obj.streams[key], obj.data[obj.data.evid == key])
                print("epi_direction: ", obj.data[obj.data.evid == key].epi_direction.values)
                print("angle: ", obj.data[obj.data.evid == key].angle.values)
                print("offset: ",obj.data[obj.data.evid == key].offset.values)
                print("snr: ",obj.data[obj.data.evid == key].snr.values)
                print("Ps: ", obj.data[obj.data.evid == key].Ps.values)
                print("Pp: ", obj.data[obj.data.evid == key].Pp.values)
                print("inclination: ", obj.data[obj.data.evid == key].inclination.values)


                plt.show()

    data = obj.data
    # data.to_csv(f"GLHS_rotation/{station}_{channels}.csv",index=False)
    data.to_csv(f"pWaveParam/{station}.csv", index=False)


# fig, ax = plt.subplots()
# divider = make_axes_locatable(ax)
# pos= ax.scatter(data["epi_direction"], data["offset"], c=data["Ps"])
# ax_histy = divider.append_axes("right", 0.6, pad=0.1, sharey=ax)
# ax_histy.yaxis.set_tick_params(labelleft=False)
# binwidth = 10
# bins = np.arange(-185, 185 + binwidth, binwidth)
# ax_histy.hist(data["offset"], bins=bins, orientation='horizontal')
# fig.colorbar(pos, ax=ax)
# ax.set_aspect('equal', adjustable='box')
# plt.show()

# print(np.c_[np.array([-0.1243848,0.99223406]),0,0])



# for key in obj.streams.keys():
#     # print(obj.streams[key].stats)
#
#     plot_3d_motion(obj.streams[key], obj.data[obj.data.evid == key])
#     plt.show()

print("Done")