import numpy as np
import obspy
import pandas as pd
import os

batches = "batch1,batch2,batch3,batch4,batch5,batch6,batch7,batch8,batch9,batch10,batch11,batch12".split(",")
batches = ["batch_all"]

# trace_path = f"../data/trace_data/Traces_{batch}/"
# save_path = "../data/Traces_main/"

save_path = "/Users/guy.bendor/Traces/StreamForPca/"
origin_file = "../data/relocated3.csv"
origins = pd.read_csv(origin_file, dtype={"evid": str})
origins.UTCtime = origins.UTCtime.apply(obspy.UTCDateTime)

for key, val in {"starttime": -2, "endtime": 2}.items():
    origins[key] = origins.UTCtime + val*60

if not os.path.exists(save_path):
    os.mkdir(save_path)

for batch in batches:
    # df = pd.read_csv(f"dataframes/data_{batch}.csv", dtype={"evid": str})
    df = pd.read_csv(f"dataframes/data_batch_combined.csv", dtype={"evid": str})
    trace_path = f"/Users/guy.bendor/Traces/StreamBatch/Traces_{batch}/"
    df.dropna(inplace=True)
    df["stream_name"] = df.evid.str.split(",").str[0]+".mseed"

    lst_col = "evid"
    r = pd.DataFrame({
          col: np.repeat(df[col].values, df.evid.str.split(",").str.len())
          for col in ["station", "stream_name"]}
        ).assign(**{lst_col: np.concatenate(df.evid.str.split(",").values)})#[df2.columns]

    new = pd.merge(r, origins, how="left", on="evid")[["station", "stream_name", "evid", "starttime", "endtime"]]

    new = new[new.stream_name.isin(os.listdir(trace_path))]


    for num, st_name in enumerate(new.stream_name.unique()):
        print(num, len(new.stream_name.unique()))
        st = obspy.read(trace_path + st_name)
        temp = new[new.stream_name == st_name]
        for name, temp_st in st._groupby("{station}{location}").items():
            if len(set((i.stats.channel for i in temp_st))) < 3:
                continue

            channel = "BH"
            if name[-2:] == "21":
                channel = "EN"
                name = name[:-2]

            elif name[-2:] == "22":
                channel = "HH"
                name = name[:-2]

            if not os.path.exists(os.path.join(save_path, name)):
                os.mkdir(os.path.join(save_path, name))

            if not os.path.exists(os.path.join(save_path, name, channel)):
                os.mkdir(os.path.join(save_path, name, channel))

            for i in range(len(temp)):
                if os.path.exists(os.path.join(save_path, name, channel, temp.iloc[i].evid + ".mseed")):
                    continue
                temp_st.write(os.path.join(save_path, name, channel, temp.iloc[i].evid + ".mseed"), format="MSEED")