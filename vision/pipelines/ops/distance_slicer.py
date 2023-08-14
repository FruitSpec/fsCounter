import os.path
from concurrent.futures import ProcessPoolExecutor
from itertools import chain
import numpy as np
import pandas as pd
from math import atan2, cos, radians, sin, sqrt
from slice_inside_frames_by_distance import slice_inside_frames

class DistanceGPS:

    def __init__(self, lon0, lat0):
        self.longitude_previous = lon0
        self.latitude_previous = lat0

    def get_distance(self, lon1, lat1):
        """
        Calculate the Haversine distance in meters, between two points.
        """

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
        return distance_meters


def extract_gnss_data(df_jz, df_gps):

    df_jz = arrange_ids(df=df_jz)

    df_gps["timestamp_gnss"] = pd.to_datetime(df_gps["timestamp"], unit="ns").dt.time
    df_gps.drop('timestamp', axis='columns', inplace=True)

    df_jz["ZED_timestamp"] = pd.to_datetime(df_jz["ZED_timestamp"], unit="ns").dt.time

    # Extract the timestamp values from both DataFrames
    zed_timestamps = df_jz['ZED_timestamp'].values
    gps_timestamps = df_gps['timestamp_gnss'].values

    # Find the indices of the last matching timestamps in df_gps
    last_indices = gps_timestamps.searchsorted(zed_timestamps, side='right')
    if np.all(last_indices == -1):
        print("No matching GPS data.")
        return None

    # Filter df_gps using the last indices
    filtered_gps = df_gps.loc[last_indices]

    # Concatenate df_merged and filtered_gps
    merged_df = pd.concat([df_jz.reset_index(drop=True), filtered_gps.reset_index(drop=True)], axis=1)

    # Filter remaining globals
    #merged_df = merged_df.query('plot!="global"')  #todo: enter?

    if len(merged_df) == 0: # all plots are global
        print("No matching GPS data.")
        return None

    # Add time difference column, between JAI and GNSS timestamps
    merged_df['JAI_timestamp'] = pd.to_datetime(merged_df['JAI_timestamp']).dt.time
    merged_df['timestamp_gnss'] = pd.to_datetime(merged_df['timestamp_gnss'], format='%H:%M:%S.%f').dt.time
    # Convert time to seconds past midnight and calculate the difference
    merged_df['time_diff_JAI_GNSS'] = merged_df['JAI_timestamp'].apply(
        lambda t: t.hour * 3600 + t.minute * 60 + t.second + t.microsecond * 1e-6) - \
                                     merged_df['timestamp_gnss'].apply(
                                         lambda t: t.hour * 3600 + t.minute * 60 + t.second + t.microsecond * 1e-6)
    merged_df['time_diff_ZED_GNSS'] = merged_df['ZED_timestamp'].apply(
        lambda t: t.hour * 3600 + t.minute * 60 + t.second + t.microsecond * 1e-6) - \
                                     merged_df['timestamp_gnss'].apply(
                                         lambda t: t.hour * 3600 + t.minute * 60 + t.second + t.microsecond * 1e-6)

    merged_df['time_diff_ZED_JAI'] = merged_df['ZED_timestamp'].apply(
        lambda t: t.hour * 3600 + t.minute * 60 + t.second + t.microsecond * 1e-6) - \
                                     merged_df['JAI_timestamp'].apply(
                                         lambda t: t.hour * 3600 + t.minute * 60 + t.second + t.microsecond * 1e-6)
    merged_df['ZED_timestamp'] = pd.to_datetime(merged_df['ZED_timestamp'].astype(str))
    merged_df['time_diff_ZED'] = merged_df['ZED_timestamp'].diff().dt.total_seconds()
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

def interploate_distance(distance):

    distance = np.array(distance)
    gps_changed = distance - np.roll(distance, 1)
    nonzero_indices = np.nonzero(gps_changed)[0]

    for i in range(1,len(nonzero_indices)):
        current_gps_change_index = nonzero_indices[i]
        previous_gps_change_index = nonzero_indices[i-1]
        delta_distance = distance[current_gps_change_index] - distance[previous_gps_change_index]
        delta_intervals = current_gps_change_index - previous_gps_change_index
        avg_step = delta_distance / delta_intervals

        for j in range(delta_intervals):
            distance[previous_gps_change_index + j] = distance[previous_gps_change_index] + (avg_step * j)

    return distance


def get_slices_vector(distances, splits):
    slices = []
    slices_index = 0
    for i in range(len(distances)):
        if distances[i] == 0:
            slices.append(slices_index)
            continue

        if i in splits:
            slices_index += 1
        slices.append(slices_index)

    return slices


def get_gps_distances(df_jz, df_gps):
    if 'longitude' not in df_jz.columns or 'latitude' not in df_jz.columns:
        df = extract_gnss_data(df_jz, df_gps)
    else:
        df = df_jz
        df = df.fillna(method='bfill')

    distances = []
    # calculate distance between two points:
    dist_detector = DistanceGPS(lon0=float(df.iloc[0]['longitude']), lat0=float(df.iloc[0]['latitude']))
    for i, row in df.iterrows():
        lon = float(row['longitude'])
        lat = float(row['latitude'])
        dist = dist_detector.get_distance(lon, lat)
        distances.append(dist)

    df['distance'] = distances
    return distances, df


def remove_adjacent(splits):
    old_splits = splits[0]
    new_splits = [old_splits[0]]
    last_split = old_splits[0]
    for split in old_splits[1:]:
        if split == last_split + 1:
            last_split = split
            continue
        else:
            new_splits.append(split)
            last_split = split

    return new_splits


def slice_frames(PATH_JZ, PATH_GPS, PATH_OUTPUT, split_range=3):
    if PATH_GPS != "":
        df_gps = read_nav_file(PATH_GPS)
    else:
        df_gps = pd.DataFrame()
    df_jz = pd.read_csv(PATH_JZ)

    # sort df_jz by 'JAI_frame_number':
    df_jz = df_jz.sort_values(by=['JAI_frame_number']).reset_index(drop=True)

    distances, df = get_gps_distances(df_jz, df_gps)
    distances = interploate_distance(distances)
    df['interpulated_dist'] = distances
    df = get_slices(df, slice_distance=split_range)

    slices_df = df[['JAI_frame_number','tree_id']]
    slices_df["start"] = 1
    slices_df["end"] = 1535
    slices_df = slices_df.rename(columns={'JAI_frame_number': 'frame_id'})
    slices_df.to_csv(PATH_OUTPUT)
    print(f'Saved {PATH_OUTPUT}')
    return slices_df, df


def arrange_ids(df):

    jai_frame_ids = np.array(list(df['JAI_frame_number']))
    zed_frame_ids = np.array(list(df['ZED_frame_number']))

    # find start index
    zeros = np.where(zed_frame_ids == 0)
    if isinstance(zeros, tuple):
        start_index = np.argmin(zed_frame_ids)
    else:
        start_index = np.max(zeros)

    jai_offset = jai_frame_ids[start_index]
    jai_frame_ids -= jai_offset

    output_z = zed_frame_ids[start_index + 1:].tolist()
    output_j = jai_frame_ids[start_index: -1].tolist()
    output_j = list(range(len(output_j)))

    output_z.sort()
    output_j.sort()

    df = df.iloc[start_index:-1]
    df["JAI_frame_number"] = output_j
    df["ZED_frame_number"] = output_z

    return df

def get_slices(df, slice_distance):
    # Initialize variables
    next_dist = slice_distance
    slice_number = 0

    # Iterate over rows
    for i, row in df.iterrows():
        # Check if distance exceeds the next distance
        if row["interpulated_dist"] >= next_dist:
            slice_number += 1
            next_dist += slice_distance  # Increase current distance by slice_distance
        df.at[i, "tree_id"] = slice_number
    return df

def slice_row(row_path, PATH_GPS):
    PATH_JZ = os.path.join(row_path, 'jaized_timestamps.csv')
    PATH_TRANSLATIONS = os.path.join(row_path, 'jai_translation.csv')
    if not os.path.exists(PATH_TRANSLATIONS):
        PATH_TRANSLATIONS = os.path.join(row_path, 'jai_translations.csv')

    path_slices_file = f"{row_path}/all_slices.csv"

    out_df, df = slice_frames(PATH_JZ, PATH_GPS, PATH_OUTPUT=path_slices_file, split_range=3)
    # save df:
    df.to_csv(f"{row_path}/gps_distance.csv")
    updated_df = slice_inside_frames(path_slices_csv=path_slices_file, path_translations_csv=PATH_TRANSLATIONS, frame_width=1535,
                                     output_path=f"{row_path}/slices.csv")

    print('Done: ', row_path)


def get_valid_row_paths(master_folder):
    paths_list = []
    for root, dirs, files in os.walk(master_folder):
        if np.all([file in files for file in ["jaized_timestamps.csv"]]):
            row_scan_path = os.path.abspath(root)
            paths_list.append(row_scan_path)
    return paths_list


def run_on_folder(master_folder, PATH_GPS, njobs=1):
    paths_list = get_valid_row_paths(master_folder)
    n = len(paths_list)
    if njobs > 1:
        with ProcessPoolExecutor(max_workers=njobs) as executor:
            res = list(executor.map(slice_row, paths_list, [PATH_GPS]*n))
    else:
        res = list(map(slice_row, paths_list, [PATH_GPS]*n))


if __name__ == "__main__":
    # PATH_JZ = r'/media/fruitspec-lab/cam175/FOWLER_1st/BLOCK700/200723/row_14/1/jaized_timestamps.csv'
    # PATH_GPS = r'/home/lihi/FruitSpec/Data/distance_slicer_problematic/row_17/200723.nav'
    # PATH_OUTPUT = r'/media/fruitspec-lab/cam175/FOWLER_1st/BLOCK700/200723/row_14/1/all_slices.csv'
    # out_df, df = slice_frames(PATH_JZ, PATH_GPS, PATH_OUTPUT, split_range=3)


    master_folder = "/media/fruitspec-lab/cam175/FOWLER_1st/BLOCK700/200723/row_15/1"
    PATH_GPS = r'/media/fruitspec-lab/cam175/FOWLER_1st/200723.nav'
    njobs = 1
    run_on_folder(master_folder, PATH_GPS, njobs=1)

    print('Done')
