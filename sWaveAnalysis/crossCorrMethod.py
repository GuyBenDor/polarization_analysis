import matplotlib.pyplot as plt
from functions.rotationCorrelation import corr_rotated
import pandas as pd



# pick_path = "/"
pick_path = f"../pickingEvents/pickFiles"
trace_path = f"/Users/guy.bendor/Traces/StreamForPca/"
save_path = "splitParam/"


corr_rotated.set_origins("../data/relocated3.csv")
corr_rotated.set_sites("../data/sites.csv")
corr_rotated.set_paths(pick_path=pick_path,trace_path=trace_path)
corr_rotated.set_half_cycle_num(3)
# corr_rotated.set_origins()

stations = {
    # "AMID": {"instrument": "EN", "fmin": 1, "fmax": 8},
    # "RSPN": {"instrument": "EN", "fmin": 1, "fmax": 8},
    # "GEM": {"instrument": "HH", "fmin": 2, "fmax": 8},
    # "GLHS": {"instrument": "HH", "fmin": 2, "fmax": 8},
    "DSHN": {"instrument": "EN", "fmin": 1, "fmax": 8},
    # "ENGV": {"instrument": "EN", "fmin": 1, "fmax": 8},
    # "HDNS": {"instrument": "EN", "fmin": 1, "fmax": 8},
    # "KDZV": {"instrument": "EN", "fmin": 1, "fmax": 8},
    # "KNHM": {"instrument": "EN", "fmin": 1, "fmax": 8},
    # "KSH0": {"instrument": "EN", "fmin": 1, "fmax": 8},
    # "LVBS": {"instrument": "EN", "fmin": 1, "fmax": 8},
    # "MGDL": {"instrument": "EN", "fmin": 1, "fmax": 8},
    # "MTLA": {"instrument": "EN", "fmin": 1, "fmax": 8},
    # "NATI": {"instrument": "EN", "fmin": 1, "fmax": 8},
    # "RMOT": {"instrument": "EN", "fmin": 1, "fmax": 8},
    # "TVR": {"instrument": "EN", "fmin": 1, "fmax": 8},
    # "ZEFT": {"instrument": "EN", "fmin": 1, "fmax": 8},
}
evlist = ["124582"]#[str(i) for i in range(124204,124209)]

obj = corr_rotated(stations, plot_fig=True, specify_events=evlist, plot_inside=False)#['123540','123541','123542','123543','123544','123545'],plot_fig=True)
df = obj.pick_df.copy()
# print(df.columns)
for station in df.station.unique():
    temp = df[df.station == station][
        ["evid", "time_delay", "corr_coef", 'angle_2d', 'Pp_2d', 'Ps_3d', 'Inclination_3d', 'angle_3d', 'Pp_3d']
    ]
    # temp.to_csv(f"{save_path}{station}.csv", index=False)
plt.show()
print("Done")