import os
import pandas as pd
from functions.picker import Figure
from functions.baseFunctions import angle, initial_bearing, haversine
import math


station = "DSHN"
channel = "EN"
# station = "NATI"
# channel = "EN"
Type =None
filter_val = [1, 8]
stream_length = [1,120]
data_path = "../data/relocated3.csv"
pick_path = f"pickFiles/"
save_path = f"pickFiles"

orig_df = pd.read_csv(data_path, dtype={"evid": str})

tr_path = f"/Users/guy.bendor/Traces/StreamForPca/{station}/{channel}/"
tr_files = os.listdir(tr_path)

Figure.set_origins(data_path)

Figure.set_filter_values(filter_val)
Figure.set_time_for_pca(0.05)
Figure.set_stream_length(stream_length)
Figure.set_pick_path(pick_path)
Figure.set_station(station)
Figure.set_karnelPickNames(1,3)
Figure.set_save_path(save_path)

df_site = pd.read_csv(f"../data/sites.csv")
site = df_site[df_site.station == station]
orig_df["epi_dist"] = orig_df.apply(lambda x: haversine(site.longitude, site.latitude, x["lon"], x["lat"]), axis=1)
pick_df = pd.read_csv(os.path.join(pick_path, f'{station}.csv'), dtype={"evid": str})

if not "evid" in pick_df.columns.values:
    pick_df.rename(columns={"event": "evid"},inplace=True)
    pick_df = pick_df.astype({'evid': 'str'})
# pick_df_s = pick_df[pick_df.Phase == "S"][["evid", "pickTime"]]
orig_df = orig_df.merge(pick_df[pick_df.Phase == "S"][["evid", "pickTime"]], how="left", on=["evid"])
# print(orig_df[orig_df.pickTime.isnull()].evid.values)

orig_df["dt"] = (
pd.to_datetime(orig_df["pickTime"], errors="coerce") - pd.to_datetime(orig_df["UTCtime"].str[:-1])
).astype("timedelta64[ms]")

orig_df["dt"] = orig_df["dt"]*1e-3 + 2
# print(orig_df.dtypes)

orig_df["dt"] = orig_df["dt"].fillna(120)
orig_df["dt"] = orig_df.dt.apply(math.ceil)
orig_df["dt"] = orig_df["dt"].astype(int)
events = orig_df.evid.values
endtimes = orig_df.dt.values
# print()

###180
for num in range(2737, len(events)):#1602

    file = events[num]
    if not os.path.exists(os.path.join(tr_path, file + ".mseed")):
        print(f"Event: {file}")
        print(f"iter #{num} does not exist")
        continue
    # file = tr_files[num]
    print(f"Event: {file}")
    print(f"iter #{num} of {len(events)}")
    print(f"distance: {round(orig_df[orig_df.evid==file].iloc[0].epi_dist)}")

    # print(f"time: {end}")
    # print([stream_length[0], endtimes[num]])
    Figure.set_stream_length([stream_length[0], endtimes[num]])
    Figure(os.path.join(tr_path, file + ".mseed"), pick_path, fig_3d=True, Type=Type)
    # plt.show()
    print()

print("Done")