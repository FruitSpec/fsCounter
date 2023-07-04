import os
import numpy as np
import pandas as pd
from math import atan2, cos, radians, sin, sqrt

class DistanceGPS:

    def __init__(self):
        self.longitude_previous = None
        self.latitude_previous = None

    def get_distance(self, lon1, lat1):
        """
        Calculate the Haversine distance in meters, between two points.
        """
        if self.latitude_previous is None or self.longitude_previous is None:
            self.longitude_previous = lon1
            self.latitude_previous = lat1
            return 0

        # Radius of the Earth in kilometers
        R = 6371.0

        # Convert degrees to radians
        lat1_rad = radians(lat1)
        lon1_rad = radians(lon1)
        lat2_rad = radians(self.latitude_previous)
        lon2_rad = radians(self.longitude_previous)

        # Differences
        delta_lat = lat2_rad - lat1_rad
        delta_lon = lon2_rad - lon1_rad

        # Haversine formula
        a = sin(delta_lat / 2) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(delta_lon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        # Distance
        distance_meters = R * c * 1000

        self.longitude_previous = lon1
        self.latitude_previous = lat1

        return distance_meters

def extract_gnss_data(df_jz, df_gps):

    df_gps["timestamp_gnss"] = pd.to_datetime(df_gps["timestamp"], unit="ns").dt.time
    df_gps.drop('timestamp', axis='columns', inplace=True)

    df_jz["ZED_timestamp"] = pd.to_datetime(df_jz["ZED_timestamp"], unit="ns").dt.time

    # Extract the timestamp values from both DataFrames
    merged_timestamps = df_jz['ZED_timestamp'].values
    gps_timestamps = df_gps['timestamp_gnss'].values

    # Find the indices of the last matching timestamps in df_gps
    last_indices = gps_timestamps.searchsorted(merged_timestamps, side='right') - 1
    if np.all(last_indices == -1):
        print("No matching GPS data.")
        return None

    # Filter df_gps using the last indices
    filtered_gps = df_gps.loc[last_indices]

    # Concatenate df_merged and filtered_gps
    merged_df = pd.concat([df_jz.reset_index(drop=True), filtered_gps.reset_index(drop=True)], axis=1)

    # Filter remaining globals
    #merged_df = merged_df.query('plot!="global"')

    if len(merged_df) == 0: # all plots are global
        print("No matching GPS data.")
        return None

    return merged_df

def read_nav_file(file_path):
    try:
        df = pd.read_csv(file_path, header = None)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None

    expected_columns = ["timestamp", "latitude", "longitude", "plot"]
    if df.iloc[0].tolist() == expected_columns:
        df.columns = df.iloc[0]
        df = df[1:]

    else:
        df.columns = expected_columns

    return df



if __name__ == "__main__":
    PATH_OUTPUT = r'/home/matans/Documents/fruitspec/sandbox/'
    PATH_GPS = "/home/matans/Documents/fruitspec/sandbox/060623.nav"
    PATH_JZ = "/home/matans/Documents/fruitspec/sandbox/jaized_timestamps.csv"

    output_dir = os.path.join(PATH_OUTPUT, 'rows_detection')
    df_gps = read_nav_file(PATH_GPS)
    df_jz = pd.read_csv(PATH_JZ)

    df = extract_gnss_data(df_jz, df_gps)

    # init distance detector:
    dist_detector = DistanceGPS()

    distances = []
    # calculate distance between two points:
    for i, row in df.iterrows():
        lon = float(row['longitude'])
        lat = float(row['latitude'])
        dist = dist_detector.get_distance(lon, lat)
        distances.append(dist)

    df['distances'] = distances


    # accumulative_distance: 19.87736697659144 meters
    # Distance: 19.719186996980007 meters