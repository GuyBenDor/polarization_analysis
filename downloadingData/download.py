import numpy as np
import os
import pandas as pd
# import re
# from functions.downloadDataFunc import site, event, get_event
# import matplotlib.pyplot as plt
from obspy.clients.fdsn import Client
# import math
# import datetime
import obspy

url = "http://82.102.143.46:8181"
client = Client(base_url=url, user='test', password='test')
channels = "HH?,EN?"

# batches = "batch1,batch2,batch3,batch4,batch5,batch6,batch7,batch8,batch9,batch10,batch11,batch12".split(",")
# batches =["batch_all"]
# batches = "batch12,batch13".split(",")
path = "dataframes/data1.csv"
save_path = "stream_data1"

def download_batch(path, save_path):
    # dat = pd.read_csv(f"dataframes/data_{batch_name}.csv",dtype={"evid": str})
    dat = pd.read_csv(path, dtype={"evid": str})
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # save_path = f"../data/trace_data/Traces_{batch_name}"

    # save_path = f"/Users/guy.bendor/Traces/StreamBatch"

    # if not os.path.exists(save_path):
    #     os.mkdir(save_path)
    print(dat)
    dat["starttime"] = dat["starttime"].apply(obspy.UTCDateTime)
    dat["endtime"] = dat["endtime"].apply(obspy.UTCDateTime)
    dat["tr_name"] = dat.evid.str.split(',').str[0]
    dat.dropna(inplace=True)

    # Creating a text file to add event ids of events that have already been downloaded
    if not os.path.exists(os.path.join(save_path, "existing_events.txt")):
        f = open(os.path.join(save_path, "existing_events.txt"), "w")
        f.close()

    # Extracting event ids of events that have already been downloaded from "existing_events.txt"
    with open(os.path.join(save_path, "existing_events.txt"), "r") as f:
        events_present = [str(int(line.split("\t")[0].split("#")[0])) for line in f.readlines()]

    names = []
    for i in range(len(dat)):
        row = dat.iloc[i]
        print(f"itr #{i}:  {row.tr_name}")
        if row.tr_name in events_present:
            continue

        mask_bool = False

        try:
            st = client.get_waveforms("*", row.station, "*", channels, row.starttime, row.endtime)
            st = st.merge()
            print(st)


            masked_names = set(
                [tr.stats.station + ";" + tr.stats.location for tr in st if isinstance(tr.data, np.ma.masked_array)])

            if masked_names != 0:
                mask_bool = True
                for mask in masked_names:
                    mask_st = st.select(station=mask.split(';')[0]).select(location=mask.split(';')[1])
                    for tr in mask_st:
                        st.remove(tr)
            stGroups = st._groupby("{station}_{location}")

            for key, val in stGroups.items():
                if len(val) != 3:
                    for tr in st.select(station=key.split("_")[0]).select(location=key.split("_")[0]):
                        st.remove(tr)

            if len(st) == 0:
                st = None
        except:
            print(f"\thas no data")
            st = None
        if st is None:
            name = row.tr_name+"#"
            if mask_bool:
                name += f"\t{','.join(list(masked_names))}"
        else:
            name = row.tr_name
            if mask_bool:
                name += f"\t{','.join(list(masked_names))}"

            stream_name = f"{row.tr_name}.mseed"
            st.write(os.path.join(save_path, stream_name), format="MSEED")


        # Writing the newly downloaded event ids into "existing_events.txt"
        with open(os.path.join(save_path, "existing_events.txt"), "a") as f:
            f.write(name + "\n")


# for batch_name in batches:#["batch11"]:#"batch7","batch8"]:#["batch1","batch2","batch3","batch4","batch5","batch6"]:
#     print(batch_name)
download_batch(path, save_path)

# with open(os.path.join(save_path, "existing_events.txt"), "r") as f:
#     events_present = sum([1 for line in f.readlines() if not "#" in line])
# print(events_present)
