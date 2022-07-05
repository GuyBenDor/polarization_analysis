import os
import pandas as pd

files = os.listdir("pickFiles")
print(files)
df = []
for file in files:
    if file == '.DS_Store':
        continue
    temp = pd.read_csv("pickFiles/"+file, dtype={"event": str, "evid": str})
    temp['station'] = file.split('.')[0]
    if 'event' in temp.columns:
        temp.rename({'event': 'evid'}, axis=1, inplace=True)
    df.append(temp)
df = pd.concat(df, ignore_index=True)
df.to_csv("pickData.csv", index=False)
print(df)