import pandas as pd

from functions.multiStation_picker import picker
import matplotlib.pyplot as plt


ev = "122727"
# ev = "122671"

# plot_picker(ev)

startEvent = 574
st_dic = {
    "SPR": {"instrument": "EN", "fmin": 1, "fmax": 8},
    # "AMID": {"instrument": "EN", "fmin": 1, "fmax": 8},
    "RSPN": {"instrument": "EN", "fmin": 1, "fmax": 8},
    # "GEM": {"instrument": "HH", "fmin": 2, "fmax": 8},
    # "GLHS": {"instrument": "HH", "fmin": 2, "fmax": 8},
    # "DSHN": {"instrument": "EN", "fmin": 1, "fmax": 8},
    "ENGV": {"instrument": "EN", "fmin": 1, "fmax": 8},
    "HDNS": {"instrument": "EN", "fmin": 1, "fmax": 8},
    # "KDZV": {"instrument": "EN", "fmin": 1, "fmax": 8},
    "KNHM": {"instrument": "EN", "fmin": 1, "fmax": 8},
    # "KSH0": {"instrument": "EN", "fmin": 1, "fmax": 8},
    # "LVBS": {"instrument": "EN", "fmin": 1, "fmax": 8},
    "MGDL": {"instrument": "EN", "fmin": 1, "fmax": 8},
    # "MTLA": {"instrument": "EN", "fmin": 1, "fmax": 8},
    # "NATI": {"instrument": "EN", "fmin": 1, "fmax": 8},
    "RMOT": {"instrument": "EN", "fmin": 1, "fmax": 8},
    "TVR": {"instrument": "EN", "fmin": 1, "fmax": 8},
    # "ZEFT": {"instrument": "EN", "fmin": 1, "fmax": 8},
}

orig_path = f"../data/relocated3.csv"
site_path = f"../data/sites.csv"
pick_path = "../pickingEvents/pickData.csv"
trace_path = "/Users/guy.bendor/Traces/StreamForPca"
save_path = "../pickingEvents/pickFiles1"

min_lat, max_lat = 32.67, 33.3
min_lon, max_lon = 35.47, 35.71

picker.set_origins(orig_path)
picker.set_sites(site_path)
picker.set_pickDf(pick_path)
picker.set_stations(st_dic)
picker.set_tracePath(trace_path)
picker.set_karnelPickNames(1, 3)
picker.set_time_for_pca(0.05)
picker.set_pca_extent(0.5, 0.2)
picker.set_stream_length([1, 20])
picker.set_single_station_path(save_path)

orig_df = pd.read_csv(orig_path, dtype={"evid": str})

orig_df = orig_df[
    (orig_df.lon>=min_lon) & (orig_df.lon<=max_lon) &
    (orig_df.lat>=min_lat) & (orig_df.lat<=max_lat)
]
# orig_df = orig_df[orig_df.epi_dist<50]
# print(orig_df.columns)
events = orig_df.evid.values
for num in range(startEvent, len(events)):#1602
    ev = events[num]
    print(f"Event: {ev}    {orig_df[orig_df.evid == ev].iloc[0].UTCtime}")
    print(f"iter #{num} of {len(events)}")
    picker(ev)
    print()

plt.show()

