import numpy as np
import math
import os
import pandas as pd
import obspy
import matplotlib.pyplot as plt
import matplotlib
import cartopy.io.shapereader as shpreader
import xarray as xr
from shapely.ops import unary_union
import cartopy.crs as ccrs
import cartopy.feature as cfeature


def arrow_dic(maxLim):
    arrows = {
        "E": [[-maxLim, 0], [-maxLim, -maxLim], [maxLim, maxLim]],
        "N": [[-maxLim, -maxLim], [-maxLim, 0], [maxLim, maxLim]],
        "Z": [[-maxLim, -maxLim], [-maxLim, -maxLim], [maxLim, 0]]
    }
    return arrows

def mat_rotation(ang, arr):
    """
    Rotates a matrix using the input angle
    :param ang: rotation angle (clockwise)
    :param arr: 2D array (x,y)
    :return: 2D array (x',y')
    """
    c, s = np.cos(np.radians(ang)), np.sin(np.radians(ang))
    R = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))
    return R.dot(arr)

def rot_mat(axis_vec,ang):
    def rotation(arr):

        Ex, Ey, Ez = axis_vec.T
        sang, cang = np.sin(ang), np.cos(ang)
        R = np.zeros([axis_vec.shape[0],3,3])

        # print(Ex*Ex*(1-cang)+cang)
        R[:,0,0] = Ex*Ex*(1-cang)+cang
        R[:,0,1] = Ex*Ey*(1-cang)-Ez*sang
        R[:,0,2] = Ex*Ez*(1-cang)+Ey*sang

        R[:,1,0] = Ey*Ex*(1-cang)+Ez*sang
        R[:,1,1] = Ey*Ey*(1-cang)+cang
        R[:,1,2] = Ey*Ez*(1-cang)-Ex*sang

        R[:,2,0] = Ez*Ex*(1-cang)-Ey*sang
        R[:,2,1] = Ez*Ey*(1-cang)+Ex*sang
        R[:,2,2] = Ez*Ez*(1-cang)+cang
        #print(R)
        #print(R.shape)
        #print(arr.shape)
        return np.inner(R,arr)

    return rotation

def make_array(x):
    if math.isnan(x):
        return []
    else:
        arr = np.zeros([2, 2])
        arr[:, 0] = [x, x]
        arr[:, 1] = [-1, 1]
        return arr[np.newaxis, ...]

def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

def sliding_window(data, size, stepsize=1, padded=False, axis=-1, copy=True):
    """
    Calculate a sliding window over a signal
    Parameters
    ----------
    data : numpy array
        The array to be slided over.
    size : int
        The sliding window size
    stepsize : int
        The sliding window stepsize. Defaults to 1.
    axis : int
        The axis to slide over. Defaults to the last axis.
    copy : bool
        Return strided array as copy to avoid sideffects when manipulating the
        output array.
    Returns
    -------
    data : numpy array
        A matrix where row in last dimension consists of one instance
        of the sliding window.
    Notes
    -----
    - Be wary of setting `copy` to `False` as undesired sideffects with the
      output values may occurr.
    Examples
    --------
    a = numpy.array([1, 2, 3, 4, 5])
    sliding_window(a, size=3)
    array([[1, 2, 3],
           [2, 3, 4],
           [3, 4, 5]])
    sliding_window(a, size=3, stepsize=2)
    array([[1, 2, 3],
           [3, 4, 5]])
    See Also
    --------
    pieces : Calculate number of pieces available by sliding
    """
    if axis >= data.ndim:
        raise ValueError(
            "Axis value out of range"
        )

    if stepsize < 1:
        raise ValueError(
            "Stepsize may not be zero or negative"
        )

    if size > data.shape[axis]:
        raise ValueError(
            "Sliding window size may not exceed size of selected axis"
        )

    shape = list(data.shape)
    shape[axis] = np.floor(data.shape[axis] / stepsize - size / stepsize + 1).astype(int)
    shape.append(size)

    strides = list(data.strides)
    strides[axis] *= stepsize
    strides.append(data.strides[axis])

    strided = np.lib.stride_tricks.as_strided(
        data, shape=shape, strides=strides
    )

    if copy:
        return strided.copy()
    else:
        return strided

def dotproduct(v1, v2):
    return sum((a * b) for a, b in zip(v1, v2))

def length(v):
    return math.sqrt(dotproduct(v, v))

def angle(v1, v2):
    return math.acos(dotproduct(v1, v2) / (length(v1) * length(v2)))

def initial_bearing(long1, lat1, long2, lat2):
    """
    :param long1: longitude of point 1 (in degrees)
    :param lat1:  latitude of point 1 (in degrees)
    :param long2: longitude of point 2 (in degrees)
    :param lat2: latitude of point 2 (in degrees)
    :return:
    The initial bearing on a sphere from point 1 to point 2.
    """
    theta = np.arctan2(np.sin(np.radians(long2 - long1)) * np.cos(np.radians(lat2)),
                       np.cos(np.radians(lat1)) * np.sin(np.radians(lat2)) -
                       np.sin(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.cos(np.radians(long2 - long1)))
    return np.degrees(theta) % 360

def angular_distance(long1, lat1, long2, lat2):
    """
    :param long1: longitude of point 1 (in degrees)
    :param lat1:  latitude of point 1 (in degrees)
    :param long2: longitude of point 2 (in degrees)
    :param lat2: latitude of point 2 (in degrees)
    :return:
    The angular distance between between point 1 and point 2
    """
    delta = np.arccos(np.sin(np.radians(lat1)) * np.sin(np.radians(lat2))
                      + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.cos(np.radians(long1 - long2)))
    return np.degrees(delta)

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points
    on the earth (specified in decimal degrees)
    :param x: an array containing [lon1,lat1,lon2,lat2]
    :return: a scaler distance in km
    """
    # convert decimal degrees to radians
    lon1, lat1, lon, lat = map(math.radians, [lon1, lat1, lon2, lat2])
    # haversine formula
    dlon = lon - lon1
    dlat = lat - lat1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))
    km = 6371 * c
    return km

def cross_track_distance(sta_lon, sta_lat, sta_BAZ, ev_lon, ev_lat, R=6371):
    """
    Computes the shortest distance, on a sphere, between an event location (ev_lon, ev_lat)
    and a BAZ direction (sta_lon, sta_lat, BAZ_direction)
    :param sta_lon: station longitude
    :param sta_lat: station latitude
    :param sta_BAZ: station Back-Azimuth direction
    :param ev_lon: event longitude
    :param ev_lat: event latitude
    :param R: Earth's radius
    :return: Distance in Km
    """
    delta_13 = angular_distance(sta_lon, sta_lat, ev_lon, ev_lat)
    bearing_13 = initial_bearing(sta_lon, sta_lat, ev_lon, ev_lat)
    dxt = np.arcsin(np.sin(np.radians(delta_13)) * np.sin(np.radians(bearing_13 - sta_BAZ))) * R
    return dxt

def heatmap(data, row_labels, col_labels, ax=None,cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)



    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(axis="x",top=False, bottom=True,# pad=20,
                   labeltop=False, labelbottom=True)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=0,#, ha="left",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    return im


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",textcolors=("k", "white"),threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = 0.5#im.norm(data.max())/2.
    # print(threshold)

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        temp=[]
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(abs(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            temp.append(text)
        texts.append(temp)

    return texts

def wavelength(i, noise_level, waveform, end_index=None,final=None):
    """ Computes the time window till the end of the stream. time consuming """
    i = int(i)

    if end_index is None:
        not_end = "i!=len(waveform)-1"
    else:
        not_end = f"i!={int(end_index)}"

    if final:
        if final == "up":
            ###   *
            ###-------
            ### *   *
            ###-------
            while (-noise_level <= waveform[i]) and eval(not_end):
                i = i + 1
            ###   *
            ###----------
            ### *   *
            ###----------
            ###       *
            while (waveform[i] <= -noise_level) and eval(not_end):
                i = i + 1
            ###   *
            ###--------------
            ### *   *    *
            ###--------------
            ###       *
            tag = "down"
            return i, tag

        if final == "down":
            ###-------
            ### *   *
            ###-------
            ###   *
            while (waveform[i] <= noise_level) and eval(not_end):
                i = i + 1
            ###   *
            ###----------
            ### *   *
            ###----------
            ###       *
            while (noise_level <= waveform[i]) and eval(not_end):
                i = i + 1
            ###       *
            ###--------------
            ### *   *    *
            ###--------------
            ###   *
            tag = "down"
            return i, tag

    if -noise_level <= waveform[i] <= noise_level:
        ###-------
        ###   *
        ###-------
        while (-noise_level <= waveform[i] <= noise_level) and eval(not_end):
            i = i + 1
        if noise_level < waveform[i]:
            ###     *
            ###-------  noise_level
            ### *
            ###------- -noise_level
            while (noise_level < waveform[i]) and eval(not_end):
                i = i + 1
            tag = "up"

        elif waveform[i] < -noise_level:
            ###-------  noise_level
            ### *
            ###------- -noise_level
            ###     *
            while (waveform[i] < -noise_level) and eval(not_end):
                i = i + 1
            tag = "down"
        else:
            tag = None

    elif waveform[i] < -noise_level:
        # -----  noise_level
        # ----- -noise_level
        # *
        while (waveform[i] < noise_level) and eval(not_end):
            i = i + 1
        #   *
        # -----  noise_level
        # ----- -noise_level
        # *
        while (-noise_level < waveform[i]) and eval(not_end):
            i = i + 1
        #   *
        # -----  noise_level
        # ----- -noise_level
        # *   *
        tag = "up"

    elif noise_level < waveform[i]:
        # *
        # -----  noise_level
        # ----- -noise_level
        while (-noise_level < waveform[i]) and eval(not_end):
            i = i + 1
        # *
        # -----  noise_level
        # ----- -noise_level
        #   *
        while (waveform[i] < noise_level) and eval(not_end):
            i = i + 1
        # *   *
        # -----  noise_level
        # ----- -noise_level
        #   *
        tag = "down"

    return i,tag

def compute_inclination(arr):
    inclinations = np.absolute(
        np.degrees(
            np.arctan(np.divide(arr[:, -1],
            np.sqrt((arr[:, :-1] ** 2).sum(axis=1))))
        )
    )
    return inclinations

def compute_pca(arr):
    arr = arr.T - arr.T.mean(0)
    U, s, V = np.linalg.svd(arr.T) #122719
    V0 = U.T[0]
    norm = 1 if U.T[0,2] < 0 else -1
    components = U.T * norm

    angles = np.round((np.degrees(np.arctan2(components[:, 0], components[:, 1]))) % 360, 1)
    inclination = np.absolute(np.degrees(np.arcsin(components[0,-1])))
    Ps = round(1 - ((s[1] + s[2]) / (2*s[0])), 2)
    Pp = round(1 - s[2] / s[1], 2)

    return {"angle": angles[0], "vec": components[0], "Ps": Ps, "Pp": Pp,
            "components": components, "angles": angles, "variance": s,"inclination":inclination}

def offset_calc(polar_ang, epi_direction):
    angs = np.array([polar_ang, epi_direction])
    dirs = np.c_[np.cos(np.radians(angs)), np.sin(np.radians(angs))]
    # x = np.cos(np.radians(angs))
    # y = np.sin(np.radians(angs))
    # offset = np.degrees(angle([x[0], y[0]], [x[1], y[1]]))
    # sign = 1 if ((0 <= angs[0] - angs[1] <= 180) or (angs[0] - angs[1] <= -180)) else -1

    # dirs = np.c_[x,y]
    offset = np.degrees(np.arccos(dirs[0] @ dirs[1]))
    offset = -offset if np.cross(dirs[0], dirs[1]) > 0 else offset
    # print(sign*offset,offset_test)
    # print()
    return offset

def convert_ang_to_xy(ang, cord):
    """
    Creates sets of x and y points representing the two directions of an ambiguous line.
    :param ang: angle of the main line (azimuth)
    :param cord: length of lines
    :return: array containing [[x,y] - azimuth,
                               [x,y]] - anti-azimuth
    """
    _x, _y = np.sin(np.radians([ang, ang + 180])), np.cos(np.radians([ang, ang + 180]))
    arr = np.array([[_x[0], _y[0]], [_x[1], _y[1]]]) * cord
    return arr

class StatsBuilder:
    def __init__(self, Stats, event):
        self.network = Stats.network
        self.station = Stats.station
        self.location = Stats.location
        self.columnNames = "[E,N,Z, delta_times, times]"
        self.starttime = np.datetime64(str(Stats.starttime)[:-1], "ms")
        self.endtime = np.datetime64(str(Stats.endtime)[:-1], "ms")
        self.sampling_rate = Stats.sampling_rate
        self.delta = np.timedelta64(int(Stats.delta * 1000), 'ms')
        self.npts = Stats.npts
        self.instrument = Stats.channel[:2] + "_"
        self.event = event

    def __str__(self):
        """
        Return better readable string representation of Stats object.
        """
        priorized_keys = ['network', 'station', 'location', "instrument", 'columnNames',
                          'starttime', 'endtime', 'sampling_rate', 'delta',
                          'npts', 'instrument', 'event']
        return self._pretty_str(priorized_keys)

    def _pretty_str(self, priorized_keys=[], min_label_length=16):
        """
        Return better readable string representation of AttribDict object.

        :type priorized_keys: list of str, optional
        :param priorized_keys: Keywords of current AttribDict which will be
            shown before all other keywords. Those keywords must exists
            otherwise an exception will be raised. Defaults to empty list.
        :type min_label_length: int, optional
        :param min_label_length: Minimum label length for keywords. Defaults
            to ``16``.
        :return: String representation of current AttribDict object.
        """
        keys = list(priorized_keys)
        # determine longest key name for alignment of all items
        try:
            i = max(max([len(k) for k in keys]), min_label_length)
        except ValueError:
            # no keys
            return ""
        pattern = "%%%ds: %%s" % (i)
        # check if keys exist
        other_keys = [k for k in keys if k not in priorized_keys]
        # priorized keys first + all other keys
        keys = priorized_keys + sorted(other_keys)
        head = [pattern % (k, self.__dict__[k]) for k in keys]
        return "\n".join(head)

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

class traces:

    def __init__(self, path, evid=None, Type=None, fmin=None, fmax=None):
        if not evid:
            eventName = path.split('/')[-1].split('.')[0]
        else:
            eventName = evid

        if fmin:
            self.fmin = fmin
        if fmax:
            self.fmax = fmax

        st = obspy.read(path)

        st.detrend(type="linear")
        st.taper(max_percentage=0.05, type="hann")
        max_start = max([tr.stats.starttime for tr in st])
        min_end = min([tr.stats.endtime for tr in st])
        st.trim(starttime=max_start, endtime=min_end)
        chan_labels = "ENZ"
        # dat_list = []
        # trace_lengths = []
        # for ch in chan_labels:
        #     tr = st.select(channel="??" + ch)[0].data
        #     dat_list.append(tr)
        #     trace_lengths.append(len(tr))

        # if len(set(trace_lengths)) != 1:
        #     st.plot()
        #     raise ValueError(f"Wrong trace lengths: {trace_lengths}")

        self.stream = st.copy()

        st.filter("bandpass", freqmin=self.fmin, freqmax=self.fmax, corners=4)



        # print(self.stream)
        self.stats = StatsBuilder(st[0].stats.copy(), eventName)
        if Type == "vel":
            if self.stats.instrument == "EN_":
                st.integrate()
        elif Type == "disp":
            if self.stats.instrument == "EN_":
                st.integrate()
            st.integrate()


        self.orig_time = self.get_orig_time(eventName)
        dtimes = np.arange(0, len(st[0])) * self.stats.delta.item().total_seconds()

        times = np.arange(self.stats.starttime,
                          self.stats.endtime + self.stats.delta,
                          self.stats.delta,
                          dtype='datetime64[ms]')

        chan_labels = "ENZ"
        arr = np.array([st.select(channel="??" + ch)[0].data for ch in chan_labels])
        a = arr.T
        b = dtimes[:, None]
        c = times.astype("float64")[:, None]
        #         print(self.stats.starttime.astype("float"),self.stats.endtime + self.stats.delta)
        #         print(c)
        #         floatTimes = times - epoch

        self.data = np.hstack((a, b, c)).T  # [E,N,Z, delta_times, times]

    @classmethod
    def set_picks(cls,path):
        files = os.listdir(path)
        dfs = []
        for file in files:
            # print(file,files)
            if '_' in file:
                continue
            # print(os.path.join(path, file),file)
            df = pd.read_csv(os.path.join(path, file), dtype={"event": str, "evid": str})
            df['station'] = file.split('.')[0]
            if 'event' in df.columns:
                df.rename({'event': 'evid'}, axis=1, inplace=True)
            dfs.append(df)
        dfs = pd.concat(dfs, ignore_index=True)
        dfs['pickTime'] = dfs.pickTime.values.astype('datetime64[ms]')
        cls.pick_df = dfs

    @classmethod
    def set_filter_limits(cls, fmin, fmax):
        cls.fmin = fmin
        cls.fmax = fmax

    @classmethod
    def set_origins(cls,path):
        df = pd.read_csv(path, dtype={"evid": str})
        df["orig_time"] = np.array(df.UTCtime.str[:-1].values,dtype="datetime64[ms]")
        cls.orig_df = df[["evid", "orig_time", "lat", "lon", "depth", "mb", "ms", "ml"]]

    def get_orig_time(self, evid):
        return self.orig_df[self.orig_df.evid == evid].iloc[0].orig_time

    def get_picks(self, evid, station):
        return self.pick_df[(self.pick_df.evid == evid) & (self.pick_df.station == station)].iloc[0].pickTime

class trace_data:

    def __init__(self, path, event, pick_file_path=None, pick_file_path_self=None, Type=None, fmin=None, fmax=None):

        for freq,val in zip(["fmin","fmax"],[1,5]):
            if not hasattr(self,freq):
                if eval(freq):
                    setattr(self,freq,eval(freq))
                else:
                    setattr(self, freq, val)

        st = obspy.read(path)

        st.detrend(type="linear")
        st.taper(max_percentage=0.05, type="hann")
        max_start = max([tr.stats.starttime for tr in st])
        min_end = min([tr.stats.endtime for tr in st])
        st.trim(starttime=max_start, endtime=min_end)
        chan_labels = "ENZ"
        # dat_list = []
        # trace_lengths = []
        # for ch in chan_labels:
        #     tr = st.select(channel="??" + ch)[0].data
        #     dat_list.append(tr)
        #     trace_lengths.append(len(tr))

        # if len(set(trace_lengths)) != 1:
        #     st.plot()
        #     raise ValueError(f"Wrong trace lengths: {trace_lengths}")

        self.stream = st.copy()

        st.filter("bandpass", freqmin=self.fmin, freqmax=self.fmax, corners=4)



        # print(self.stream)
        self.stats = StatsBuilder(st[0].stats.copy(), event)
        if Type == "vel":
            if self.stats.instrument == "EN_":
                st.integrate()
        elif Type == "disp":
            if self.stats.instrument == "EN_":
                st.integrate()
            st.integrate()
        # print(event)
        self.orig_time = self.get_orig_time(event)
        dtimes = np.arange(0, len(st[0])) * self.stats.delta.item().total_seconds()

        times = np.arange(self.stats.starttime,
                          self.stats.endtime + self.stats.delta,
                          self.stats.delta,
                          dtype='datetime64[ms]')

        chan_labels = "ENZ"
        arr = np.array([st.select(channel="??" + ch)[0].data for ch in chan_labels])
        a = arr.T
        b = dtimes[:, None]
        c = times.astype("float64")[:, None]
        #         print(self.stats.starttime.astype("float"),self.stats.endtime + self.stats.delta)
        #         print(c)
        #         floatTimes = times - epoch

        self.data = np.hstack((a, b, c)).T  # [E,N,Z, delta_times, times]

        # self._pick_path = pick_file_path
        self.picks = {}
        if pick_file_path:
            self.read_picks(pick_file_path)
        if pick_file_path_self:
            self.read_picks_from_self(pick_file_path_self)


    def read_picks(self, path):

        df = pd.read_csv(path, dtype={"tr_id": str, "picktime": str})
        times = np.array(df.picktime.tolist(), dtype="datetime64[ms]")
        errors = np.array((1000 * df.errors).tolist()).astype("timedelta64[ms]")

        df["pickTime_floats"] = times.astype("float")
        df["errors_deltas"] = errors
        df = df[(df.pickTime_floats >= self.stats.starttime.astype("float")) &
                (df.pickTime_floats <= self.stats.endtime.astype("float"))]
        if len(df) == 0:
            return

        P_picks = df[df.iphase == "P"][["pickTime_floats", "errors_deltas"]].to_numpy().T
        S_picks = df[df.iphase == "S"][["pickTime_floats", "errors_deltas"]].to_numpy().T
        dic = {
            "P": {"picks": P_picks[0], "errors": P_picks[1]},
            "S": {"picks": S_picks[0], "errors": S_picks[1]}
        }
        self.picks.update(dic)

    def read_picks_from_self(self, path):

        df = pd.read_csv(path, dtype={"event": str, "pickTime": str})
        times = np.array(df.pickTime.tolist(), dtype="datetime64[ms]")

        df["pickTime_floats"] = times.astype("float")
        # errors = np.array((1000 * df.errors).tolist()).astype("timedelta64[ms]")

        df = df[(df.pickTime_floats >= self.stats.starttime.astype("float")) &
                (df.pickTime_floats <= self.stats.endtime.astype("float"))]
        if len(df) == 0:
            return

        P_picks = df[df.Phase == "P"][["pickTime_floats"]].to_numpy().T
        S_picks = df[df.Phase == "S"][["pickTime_floats"]].to_numpy().T
        # print(P_picks)
        dic = {
            "P": {"picks": P_picks[0]},
            "S": {"picks": S_picks[0]}
        }
        self.picks.update(dic)

    @classmethod
    def set_filter_limits(cls,fmin,fmax):
        cls.fmin=fmin
        cls.fmax=fmax

    def get_orig_time(self, evid):
        return self.orig_df[self.orig_df.evid == evid].iloc[0].orig_time

    @classmethod
    def set_origins(cls,path):
        df = pd.read_csv(path,dtype={"evid":str})
        df["orig_time"] = np.array(df.UTCtime.str[:-1].values,dtype="datetime64[ms]")
        cls.orig_df = df[["evid", "orig_time", "lat", "lon", "depth", "mb", "ms", "ml"]]

class tracesMan(traces):
    def __init__(self, path, evid=None, Type=None, fmin=None, fmax=None):
        super().__init__(path)

    def get_orig_time(self, evid):
        return self.df[self.df.evid == evid].iloc[0].orig_time

    def get_picks(self, evid, station):
        return self.df[(self.df.evid == evid) & (self.df.station == station)].iloc[0].pickTime

    @classmethod
    def set_data(cls, path):
        df = pd.read_csv(path, dtype={"evid": str})
        df["orig_time"] = np.array(df.UTCtime.str[:-1].values, dtype="datetime64[ms]")
        df['pickTime'] = df.pickTime.values.astype('datetime64[ms]')
        cls.df = df

def get_map_layers(fault,water,top,border):
    da = xr.open_dataset(top)
    f = cfeature.ShapelyFeature(shpreader.Reader(fault).geometries(), ccrs.PlateCarree())
    w = cfeature.ShapelyFeature(shpreader.Reader(water).geometries(), ccrs.PlateCarree())
    w = unary_union(list(w.geometries())).geoms
    b = cfeature.ShapelyFeature(shpreader.Reader(border).geometries(), ccrs.PlateCarree())
    return f, w, da, b


class stations:
    def __init__(self):
        pass

    @classmethod
    def set_sta_path(cls, sta_path):
        df = pd.read_csv(sta_path)
        df.rename(columns={"latitude": "lat", "longitude": "lon", "elevation": "elv"}, inplace=True)
        cls.sta_df = df


class events:
    def __init__(self):
        pass

    @classmethod
    def set_error_min_max(cls, emin, emax):
        cls.emin = emin
        cls.emax = emax

    @classmethod
    def remove_quality_matches(cls, qual_values):
        print("Quality_matches should be defined before 'ev_paths' is set.\n"
              "Should be a list of code values to remove\n"
              "\t(currently [0: well constrained, 1: weekly constrained])")
        cls.qual_values = qual_values

    @classmethod
    def set_ev_paths(cls, orig_path, **kwargs):
        """

        :param orig_path:
        :param kwargs: optional: pick_path, trace_path
        :return:
        """
        orig = pd.read_csv(orig_path, dtype={"evid": str})
        orig["UTCtime"] = orig["UTCtime"].apply(obspy.UTCDateTime)
        orig.drop(["prefor", "time", "nass", "ndef", "Unknown"], axis=1, inplace=True)
        orig["mag"] = orig.ms
        orig.loc[orig.mag.isna(), "mag"] = orig.loc[orig.mag.isna(), "ml"]
        orig.loc[orig.mag.isna(), "mag"] = orig.loc[orig.mag.isna(), "mb"]
        orig.rename(columns={"UTCtime": "originTime"}, inplace=True)
        if "pick_path" in kwargs:
            pick = pd.read_csv(kwargs['pick_path'], dtype={"tr_id": str})

            fill_error_val = cls.emax if hasattr(cls, "emax") else 0
            pick.fillna(value={"errors":fill_error_val},inplace=True)

            pick["picktime"] = pick["picktime"].apply(obspy.UTCDateTime)
            pick.rename(columns={"picktime": "pickTime", "tr_id": "evid", "errors": "error"}, inplace=True)

            if hasattr(cls, "emin"):
                # pick['error'] = np.where((pick.error < cls.emin), cls.emin, pick.error)  # <<<<<<<<<<<<<<<<<<<<<
                pick.loc[pick.error < cls.emin, "error"] = cls.emin
            if hasattr(cls, "emax"):
                pick.loc[pick.error > cls.emax, "error"] = cls.emax
                # pick['error'] = np.where((pick.error > cls.emax), cls.emax, pick.error)  # <<<<<<<<<<<<<<<<<<<<<
            errors = pick['error'].tolist()
            errors.sort()
            print(f"Top 10 max error values: {[ '%.2f' % err for err in errors[-10:]]}")
            if hasattr(cls, "qual_values"):
                pick = pick[~pick.match_quality.isin(cls.qual_values)]

            orig = orig.merge(pick, how="inner", on="evid")
        cls.ev_df = orig
        if "trace_path" in kwargs:
            cls._trace_path = kwargs['trace_path']

class sta_info(stations):
    def __init__(self, name):
        self._name = name
        self.set_sta_data()

    def set_sta_data(self):
        _row = self.sta_df[self.sta_df.station == self._name].iloc[0]
        atters = ["lat", "lon", "elv"]
        for att in atters:
            setattr(self, att, _row[att])

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value
        self.set_sta_data()
