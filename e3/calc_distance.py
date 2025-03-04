import sys
import xml.etree.ElementTree as ET
import pandas as pd
import numpy as np
# from numpy import sin, cos, arctan, sqrt
from math import radians, cos, sin, asin, sqrt, atan2
from pykalman import KalmanFilter
from xml.dom.minidom import getDOMImplementation

def read_gpx(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    # ns = {'default': 'http://www.topografix.com/GPX/1/0'}
    
    latitudes = []
    longitudes = []
    
    for trkpt in root.iter('{http://www.topografix.com/GPX/1/0}trkpt'):
        latitudes.append(float(trkpt.get('lat')))
        longitudes.append(float(trkpt.get('lon')))
    
    gpxdf = pd.DataFrame(data={'lat': latitudes, 'lon': longitudes})
    return gpxdf

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000  # Radius of Earth in kilometers
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    # print(type(lat1))
    dislat = lat2 - lat1
    dislon = lon2 - lon1
    # print(type(dislat))
    a = sin(dislat/2) * sin(dislat/2) + cos(lat1) * cos(lat2) * sin(dislon/2) * sin(dislon/2)
    # print(a)
    # a = np.array(a)
    # print(a)
    c = 2 * asin(sqrt(a))
    print(c)
    return R * c

def distance(df):
    df_shifted = df.shift(-1)
    # print(df_shifted)
    df_shifted = df_shifted[:-1]  # Drop last row (NaN after shift)
    # print(df_shifted)
    dists = []
    # dists = [haversine(df.iloc[i]['lat'], df.iloc[i]['lon'],
    #                 df_shifted.iloc[i]['lat'], df_shifted.iloc[i]['lon'])
    #         for i in range(len(df_shifted))]
    for i in range(len(df_shifted)):
        distance = haversine(df.iloc[i]['lat'], df.iloc[i]['lon'], df_shifted.iloc[i]['lat'], df_shifted.iloc[i]['lon'])
        dists.append(distance)


    return sum(dists)

def smooth(df):
    initial_state = df.iloc[0]
    observation_covariance = np.diag([20e-6, 20e-6])
    transition_covariance = np.diag([5e-6, 5e-6]) 
    # print(observation_covariance)
    # print(transition_covariance)
    transition_matrices = np.eye(2)
    
    kf = KalmanFilter(initial_state_mean=initial_state,
                       observation_covariance=observation_covariance,
                       transition_covariance=transition_covariance,
                       transition_matrices=transition_matrices)
    
    smoothed_state_means, _ = kf.smooth(df.values)
    
    return pd.DataFrame(smoothed_state_means, columns=['lat', 'lon'])

def append_trkpt(pt, trkseg, doc):
    trkpt = doc.createElement('trkpt')
    trkpt.setAttribute('lat', '%.8f' % (pt['lat']))
    trkpt.setAttribute('lon', '%.8f' % (pt['lon']))
    trkseg.appendChild(trkpt)

def output_gpx(points, output_filename):
    """
    Output a GPX file with latitude and longitude from the points DataFrame.
    """
    doc = getDOMImplementation().createDocument(None, 'gpx', None)
    trk = doc.createElement('trk')
    doc.documentElement.appendChild(trk)
    trkseg = doc.createElement('trkseg')
    trk.appendChild(trkseg)
    
    points.apply(append_trkpt, axis=1, trkseg=trkseg, doc=doc)
    
    with open(output_filename, 'w') as fh:
        doc.writexml(fh, indent=' ')

def main():
    if len(sys.argv) < 2:
        print("File Usage: python3 calc_distance.py <file.gpx>")
        sys.exit(1)
    
    points = read_gpx(sys.argv[1])
    print('Unfiltered distance: %0.2f' % distance(points))
    
    smoothed_points = smooth(points)
    print('Filtered distance: %0.2f' % distance(smoothed_points))
    
    output_gpx(smoothed_points, 'out2.gpx')

if __name__ == "__main__":
    main()
