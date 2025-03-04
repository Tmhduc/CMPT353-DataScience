import pandas as pd

df = pd.read_csv('totals.csv').set_index(keys=['name'])
df_count = pd.read_csv('counts.csv').set_index(keys=['name'])

# print(df.to_string())

lowest_precipitation_city = df.sum(axis=1).idxmin()
print("City with the lowest total precipitation:")
print(lowest_precipitation_city)

# print(df.sum(axis=0))

precipitation_per_month = df.sum(axis=0) / df_count.sum(axis=0)
print("Average precipitation per month:")
print(precipitation_per_month)

average_precipitation_per_city = df.sum(axis=1) / df_count.sum(axis=1)
print("Average precipitation per city:")
print(average_precipitation_per_city)