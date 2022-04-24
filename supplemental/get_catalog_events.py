import numpy as np
import os
import pandas as pd
import obspy
import re
# from fun_dir.download_func import site, event, get_event
import matplotlib.pyplot as plt
from obspy.clients.fdsn import Client
import math
import datetime
import obspy

url = "http://82.102.143.46:8181"
# url = "http://172.16.46.102:8181"
client = Client(base_url=url, user='test', password='test')

# 9/28/2020

t1 = obspy.UTCDateTime("2020-09-28T00:00:00")
t2 = obspy.UTCDateTime("2022-01-12T03:00:00")
cat = client.get_events(starttime=t1, endtime=t2,eventtype="earthquake",includearrivals=True)
cat.write("catalog28092020_12012022.xml", format="QUAKEML")

print(cat)