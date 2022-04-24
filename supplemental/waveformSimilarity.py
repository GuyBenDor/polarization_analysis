import obspy
import numpy as np
import pandas as pd
import os
from obspy.signal.cross_correlation import xcorr_3c, correlate_stream_template,correlate_template
from functions.baseFunctions import find_nearest
# from scipy.signal import correlate, correlation_lags
# from functions.rotationCorrelation import slide_correlation
# from functions.baseFunctions import sliding_window

data_path = "../data/relocated3.csv"
save_path = "similarities"

origDf = pd.read_csv(data_path, dtype={"evid": str})
origDf["UTCtime"] = origDf.UTCtime.apply(obspy.UTCDateTime)
# origDf["endtime"] = origDf["UTCtime"]+2*60
dt = 0.8

# print(origDf.UTCtime)

ev_name = "123063"

station = "RSPN"
channel = "EN"
pickDf = pd.read_csv(f'../pickingEvents/pickFiles/{station}.csv', dtype={'evid': str})
pickDf["pickTime"] = pickDf.pickTime.apply(obspy.UTCDateTime)
pickDf = pickDf[pickDf.Phase == "S"]
pickDf["endTime"] = pickDf["pickTime"] + dt
pickDf = pickDf.merge(origDf, on="evid")
pickDf = pickDf.sample(frac=1).reset_index(drop=True)
# print(pickDf)

tr_path = f"/Users/guy.bendor/Traces/StreamForPca/{station}/{channel}/"

# tr_files = os.listdir(tr_path)

def coeffs(arr):
    return np.corrcoef(arr)

traces = {}
for num in range(len(pickDf)):
    row = pickDf.iloc[num]
    st = obspy.read(f"{tr_path}{row.evid}.mseed", dtype=np.float64,starttime=row.UTCtime)
    st.detrend(type="linear")
    st.taper(max_percentage=0.05, type="hann")
    # st.trim(starttime=row.pickTime, endtime=row.endTime)
    # print(st.data)
    # st.filter("bandpass", freqmin=self.fmin, freqmax=self.fmax, corners=4)
    st = st.filter("bandpass", freqmin=1, freqmax=8, corners=4)
    traces[row.evid] = {
        "st": st,
        "data": np.array([tr.data for tr in st]),
        "slice": np.array([tr.data for tr in st.slice(starttime=row.pickTime, endtime=row.endTime)]),
        "pick_ind": find_nearest(
            np.datetime64(st[0].stats.starttime) + np.arange(0, len(st[0])) * np.timedelta64(
                int(st[0].stats.delta * 1000), "ms"),np.datetime64(row.pickTime)
        )

    }


lt = len(traces.keys())
groups = []
groups_evids = [[],[],[]]
j = 0
while len(traces.keys()) > 0:
    print(j, len(traces.keys()))
    evid = list(traces.keys())[0]
    row = pickDf[pickDf.evid == evid].iloc[0]
    template = traces[evid]["slice"]
    group = []
    group_evid = []
    temp = np.zeros([len(traces.keys()), 2])
    # template.plot()

    for num, (key, val) in enumerate(traces.items()):
        sig = []
        for i in range(3):
            cc = correlate_template(val["data"][i], template[i], mode="valid")
            sig.append(cc)
        sig = np.array(sig).sum(0)
        temp[num, 0] = np.abs(sig).argmax()
        temp[num, 1] = np.abs(sig).max()/3
    remove_inds = np.argwhere(np.abs(temp[:, 1]) >= 0.95)
    for ind in remove_inds[::-1]:
        key = list(traces.keys())[ind[0]]
        group.append(traces[key])
        group_evid.append(key)
        del traces[key]
    group_par = list(temp[remove_inds.T[0], :])
    group_evid, group = zip(*sorted(zip(group_evid, group)))
    groups.append(group)
    groups_evids[0].append(group_evid)
    groups_evids[1].append(group_par)
    groups_evids[2].append(evid)
    j += 1

if not os.path.exists(save_path):
    os.makedirs(save_path)
import pickle
with open(f'{save_path}/{station}_set3.pkl', 'wb') as f:
   pickle.dump(groups_evids, f)
# print(groups)
print(len(groups), lt)
# print(indexes)
# print(values)
# print(temp)
# print(len(traces.keys()))



# print(template[0].stats)