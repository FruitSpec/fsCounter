import os
import numpy as np
import pandas as pd
from math import atan2, cos, radians, sin, sqrt
from vision.pipelines.ops.frame_loader import arrange_ids
from vision.pipelines.ops.slice_inside_frames import slice_inside_frames

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

    #df_jz = arrange_ids(df=df_jz)

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
    df = extract_gnss_data(df_jz, df_gps)

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

def slices_to_df(df, output_path):

    slices_df = df[['JAI_frame_number', 'tree_id']]
    slices_df["start"] = 1
    slices_df["end"] = 1535
    slices_df = slices_df.rename(columns={'JAI_frame_number': 'frame_id'})

    if output_path is not None:
        slices_df.to_csv(output_path)
        print(f'Saved {output_path}')

    return slices_df


def slice_frames(PATH_JZ, PATH_GPS, output_path=None, split_range=3):
    df_gps = read_nav_file(PATH_GPS)
    df_jz = pd.read_csv(PATH_JZ)

    df_jz = update_df_ids(df_jz)
    distances, df = get_gps_distances(df_jz, df_gps)
    distances = interploate_distance(distances)

    splits_tuple = np.where(distances % split_range < 0.1)
    splits = remove_adjacent(splits_tuple)
    slices = get_slices_vector(distances, splits)

    df['interpulated_dist'] = distances
    df['tree_id'] = slices

    slices_df = slices_to_df(df, output_path)

    return slices_df, df

def update_df_ids(df_jz):
    output_z, output_j, start_index = arrange_ids(df_jz['JAI_frame_number'], df_jz['ZED_frame_number'], True)
    df_jz = df_jz.iloc[start_index:-1]
    df_jz["JAI_frame_number"] = output_j
    df_jz["ZED_frame_number"] = output_z

    return df_jz

def slice_row(row_path, nav_path):

    jz_path = os.path.join(row_path, 'jaized_timestamps.csv')
    translations_path = os.path.join(row_path, "jai_translation.csv")

    t_df = pd.read_csv(translations_path)


    slices_df, df = slice_frames(jz_path, nav_path, output_path=None, split_range=3)
    updated_df = slice_inside_frames(slices_df, t_df, 1536, output_path=None)

    return updated_df



# def arrange_ids(df):
#
#     jai_frame_ids = np.array(list(df['JAI_frame_number']))
#     zed_frame_ids = np.array(list(df['ZED_frame_number']))
#
#     # find start index
#     zeros = np.where(zed_frame_ids == 0)
#     if isinstance(zeros, tuple):
#         start_index = np.argmin(zed_frame_ids)
#     else:
#         start_index = np.max(zeros)
#
#     jai_offset = jai_frame_ids[start_index]
#     jai_frame_ids -= jai_offset
#
#     output_z = zed_frame_ids[start_index + 1:].tolist()
#     output_j = jai_frame_ids[start_index: -1].tolist()
#     output_j = list(range(len(output_j)))
#
#     output_z.sort()
#     output_j.sort()
#
#     df = df.iloc[start_index:-1]
#     df["JAI_frame_number"] = output_j
#     df["ZED_frame_number"] = output_z
#
#     return df




if __name__ == "__main__":

    row_path = r'/media/matans/My Book/FruitSpec/Customers_data/Fowler/daily/OLIVER12/180723/row_10/1'
    nav_path =  r'/media/matans/My Book/FruitSpec/Customers_data/Fowler/nav/180723.nav'
    jz_path = os.path.join(row_path, 'jaized_timestamps.csv')
    translations_path = os.path.join(row_path, "jai_translations.csv")

    t_df = pd.read_csv(translations_path)
    path_slices_file = f"{row_path}/slices.csv"

    slices_df, df = slice_frames(jz_path, nav_path, output_path=path_slices_file, split_range=3)
    updated_df = slice_inside_frames(slices_df, t_df, 1536, output_path=None)
    #updated_df = slice_inside_frames(path_slices_csv=path_slices_file, path_translations_csv=PATH_TRANSLATIONS, frame_width=1535,
    #                                 output_path=f"{row_path}/all_slices.csv")
    print('Done')
