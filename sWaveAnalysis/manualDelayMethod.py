import os
import pandas as pd
from functions.manualDelay import Figure
from functions.baseFunctions import haversine


station = "RSPN"
channel = "EN"
# station = "NATI"
# channel = "EN"
Type =None
filter_val = [1, 8]
stream_length = [0., 0.25]
data_path = "/Users/guy.bendor/Jupyter Notebooks/files_for_linoy/data_0410.csv"
# pick_path = f"/Users/guy.bendor/PycharmProjects/ShearWave_splitting/picking_events/pickFiles2"
# # save_path = f"/Users/guy.bendor/PycharmProjects/ShearWave_splitting/picking_events/pickFiles2"
# rotation_path = f"/Users/guy.bendor/PycharmProjects/ShearWave_splitting/picking_events/automated_split_param4"

Figure.set_data(data_path)
Figure.set_filter_values(filter_val)
Figure.set_stream_length(stream_length)
Figure.set_save_path(data_path)
# Figure.set_pick_path(pick_path)
# Figure.set_rotation_path(rotation_path)

# orig_df = pd.read_csv(data_path, dtype={"evid": str})
# df_site = pd.read_csv(f"../data/main_data_files/sites.csv")
# site = df_site[df_site.station == station]
# orig_df["epi_dist"] = orig_df.apply(lambda x: haversine(site.longitude,site.latitude, x["lon"], x["lat"]),axis=1)

event_df = pd.read_csv(data_path, dtype={"evid": str})
event_df = event_df[event_df.station==station]
# print(event_df)

tr_path = f"/Users/guy.bendor/Traces/StreamForPca/{station}/{channel}/"
# tr_files = os.listdir(tr_path)
# print(event_df)
events = event_df.evid.unique()
for num in range(0, 100):#len(events)):

    file = events[num]
    if not os.path.exists(os.path.join(tr_path, file + ".mseed")):
        print(f"Event: {file}")
        print(f"iter #{num} does not exist")
        continue
    # file = tr_files[num]
    print(f"Event: {file}")
    print(f"iter #{num} of {len(events)}")
    # print(f"distance: {round(orig_df[orig_df.evid==file].iloc[0].epi_dist)}")
    # print(f"time: {end}")
    Figure(os.path.join(tr_path, file + ".mseed"), Type=Type, rotation="2d", fig_3d=True)
    # plt.show()