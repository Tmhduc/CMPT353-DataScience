import numpy as np
import pandas as pd
import sys
from math import radians, cos, sin, asin, sqrt
import matplotlib.pyplot as plt

# Loading from json file
def load_json(stations_file):
    return pd.read_json(stations_file, lines=True)

# Load CSV data
def read_city(city_file):
    cities = pd.read_csv(city_file)
    cities = cities.dropna(subset=["population", "area"])
    cities["areas_km2"] = cities['area'] / 1e6
    cities = cities[cities['areas_km2'] <= 100000]
    cities['density'] = cities['population'] / cities['areas_km2'] 
    return cities

# Haversine formula to calculate distance
def haversine(lat1, lon1, lat2, lon2):
    R = 6371000 # Radius of Earth in kilometers
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dislat = lat2 - lat1
    dislon = lon2 - lon1
    a = sin(dislat/2) * sin(dislat/2) + cos(lat1) * cos(lat2) * sin(dislon/2) * sin(dislon/2)
    c = 2 * asin(sqrt(a))
    return R * c

def cal_distance(city, stations):
    distances = stations.apply(lambda station : haversine(
        city['latitude'], city['longitude'],
        station['latitude'], station['longitude']
    ), axis=1)
    return stations.iloc[distances.idxmin()]['avg_tmax'] / 10

def best_tmax(cities, stations):
    cities['temperature'] = cities.apply(cal_distance, stations=stations, axis=1)
    return cities[['name', 'density', 'temperature']]

def scatterplot(cities, output):
    plt.figure(figsize=(8,6))
    plt.scatter(cities['density'], cities['temperature'], alpha=0.5)
    plt.xlabel('Population density (people/km2)')
    plt.ylabel('Avg Max Temperature (C)')
    plt.title('Correlation between Population Density and Temperature')
    plt.xscale('log')
    plt.savefig(output)
    plt.show()
def main():
    stationsdf = load_json(sys.argv[1])
    citiesdf = read_city(sys.argv[2])
    output_file = sys.argv[3]
    result = best_tmax(citiesdf, stationsdf)
    
    scatterplot(result, output_file)
    
if __name__ == "__main__":
    main()