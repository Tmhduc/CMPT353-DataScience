import numpy as np

# Load the data
data = np.load('monthdata.npz')
totals = data['totals']
counts = data['counts']

total_precipitation_city = totals.sum(axis=1)
lowest_city = np.argmin(total_precipitation_city)
print("City with the lowest total precipitation (row index):", lowest_city)