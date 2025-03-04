import numpy as np

# Load the data
data = np.load('monthdata.npz')
#print(type(data))
totals = data['totals']
# print(totals)
counts = data['counts']
# print(counts)

total_precipitation_city = totals.sum(axis=1)
lowest_city = np.argmin(total_precipitation_city)
print("Row with lowest total precipitation:\n", lowest_city)

average_precipitation_per_month = totals.sum(axis=0) / counts.sum(axis=0)
print("Average precipitation per month:\n", average_precipitation_per_month)

average_precipitation_per_city = totals.sum(axis=1) / counts.sum(axis=1)
print("Average precipitation per city:\n", average_precipitation_per_city)

quarters_totals = totals.reshape(totals.shape[0], 4, 3).sum(axis=2)
print("Total precipitation per quarter for each city:")
print(quarters_totals)