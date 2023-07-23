import os
import numpy as np
import pandas as pd
from math import atan2, cos, radians, sin, sqrt
import math
import cv2

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

    # # Add time difference column, between JAI and GNSS timestamps
    # merged_df['JAI_timestamp'] = pd.to_datetime(merged_df['JAI_timestamp']).dt.time
    # merged_df['timestamp_gnss'] = pd.to_datetime(merged_df['timestamp_gnss'], format='%H:%M:%S.%f').dt.time
    #
    # # Convert time to seconds past midnight and calculate the difference
    # merged_df['time_diff_JAI_GNSS'] = merged_df['JAI_timestamp'].apply(
    #     lambda t: t.hour * 3600 + t.minute * 60 + t.second + t.microsecond * 1e-6) - \
    #                                  merged_df['timestamp_gnss'].apply(
    #                                      lambda t: t.hour * 3600 + t.minute * 60 + t.second + t.microsecond * 1e-6)
    # merged_df['time_diff_ZED_GNSS'] = merged_df['ZED_timestamp'].apply(
    #     lambda t: t.hour * 3600 + t.minute * 60 + t.second + t.microsecond * 1e-6) - \
    #                                  merged_df['timestamp_gnss'].apply(
    #                                      lambda t: t.hour * 3600 + t.minute * 60 + t.second + t.microsecond * 1e-6)
    #
    # merged_df['time_diff_ZED_JAI'] = merged_df['ZED_timestamp'].apply(
    #     lambda t: t.hour * 3600 + t.minute * 60 + t.second + t.microsecond * 1e-6) - \
    #                                  merged_df['JAI_timestamp'].apply(
    #                                      lambda t: t.hour * 3600 + t.minute * 60 + t.second + t.microsecond * 1e-6)
    # # add time difference ZED:
    # merged_df['ZED_timestamp'] = pd.to_datetime(merged_df['ZED_timestamp'].astype(str))
    # merged_df['time_diff_ZED'] = merged_df['ZED_timestamp'].diff().dt.total_seconds()
    #
    # # add time difference gnss:
    # merged_df['timestamp_gnss'] = pd.to_datetime(merged_df['timestamp_gnss'].astype(str))
    # merged_df['time_diff_gnss'] = merged_df['timestamp_gnss'].diff().dt.total_seconds()

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


def slice_frames(PATH_JZ, PATH_GPS, PATH_OUTPUT, split_range):
    df_gps = read_nav_file(PATH_GPS)
    df_jz = pd.read_csv(PATH_JZ)


    distances, df = get_gps_distances(df_jz, df_gps)
    distances = interploate_distance(distances)
    df['interpulated_dist'] = distances
    df = get_frame_numbers(df, slice_distance=split_range)

    df.to_csv(PATH_OUTPUT)
    print(f'Saved {PATH_OUTPUT}')
    return df


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


def get_frame_numbers(df, slice_distance):
    # Initialize variables
    current_dist = slice_distance / 2
    df["get_frame"] = False

    # Iterate over rows
    for i, row in df.iterrows():
        # Check if distance exceeds the current distance
        if row["interpulated_dist"] >= current_dist:
            df.at[i, "get_frame"] = True
            current_dist += slice_distance  # Increase current distance by slice_distance

    return df


def calculate_Field_Of_View(depth, sensor_width=7.07, sensor_height=5.30, focal_length=6):
    '''
    1. Calculate the horizontal and vertical Angle of View (AoV) in radians
    2. Calculate the horizontal and vertical Fields of View (FoV) at the given depth.

    jai_sensor_width_mm = 7.07
    jai_sensor_height_mm = 5.30
    jai_focal_length_mm = 6
    '''

    hAoV = 2 * math.atan(sensor_width / (2 * focal_length))
    vAoV = 2 * math.atan(sensor_height / (2 * focal_length))

    hFOV = 2 * depth * math.tan(hAoV / 2)
    vFOV = 2 * depth * math.tan(vAoV / 2)

    return hFOV, vFOV

def get_frames_df(path_JZ, PATH_GPS, output_path, depth_meters):
    hFOV_meters, vFOV_meters = calculate_Field_Of_View(depth_meters)
    df = slice_frames(path_JZ, PATH_GPS, PATH_OUTPUT=output_path, split_range=vFOV_meters * 0.9)
    return df

if __name__ == "__main__":
    row_path = r'/home/fruitspec-lab-3/FruitSpec/Data/grapes/USXXXX/GRAPES/JACFAM/204401XX/180723/row_5/1'
    PATH_GPS = r'/home/fruitspec-lab-3/FruitSpec/Data/grapes/USXXXX/GRAPES/JACFAM/204401XX/180723.nav'
    DEPTH_METERS = 1
    FOV_CORRECTION = 0.9

    path_video = os.path.join(row_path, 'Result_FSI.mkv')
    path_JZ = os.path.join(row_path, 'jaized_timestamps.csv')
    # path_slices_file = f"{row_path}/gps_jai_zed.csv"

    df = get_frames_df(path_JZ, PATH_GPS, row_path, DEPTH_METERS)

    # itterate video:
    cap = cv2.VideoCapture(path_video)
    concatenated_img = None
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1  # Get the next frame ID
            # Filter dataframe for the current frame ID
            frame_data = df[df['JAI_frame_number'] == frame_id]
            # rotate image anticlockwise
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
            # crop horizontally the image, to have the second third of the image:
            frame_cropped = frame[int(frame.shape[1] / 4):int(frame.shape[1] * 2 / 4), :]
            # resize frame_cropped to 1/3 of the original size:
            frame = cv2.resize(frame_cropped, (int(frame_cropped.shape[1] / 4), int(frame_cropped.shape[0] / 4)))
            # add text - frame_id:
            cv2.putText(frame, f"Frame: {frame_id}", (10, 30), cv2.FONT_HERSHEY_DUPLEX, 1, (0, 0, 0), 2)
            # separate the frames by horizontal black line:
            cv2.line(frame, (0, 0), (frame.shape[1], 0), (0, 0, 0), 5)

            # concatenate the images:
            concatenated_img = cv2.vconcat([concatenated_img, frame]) if concatenated_img is not None else frame
            if frame_data["get_frame"].values[0]:
                cv2.imshow("frame", concatenated_img)
                cv2.waitKey(0)
                concatenated_img = None
                concatenated_img = frame


        else:
            break

    print('ok')
    # from slice_inside_frames_by_distance import slice_inside_frames
    #
    # # row_path = r'/media/fruitspec-lab/cam175/USXXXX_citrus_lihi_temp/MAZMANI2/170723/row_20/1'
    # # PATH_GPS =  r'/media/fruitspec-lab/cam175/USXXXX_citrus_lihi_temp/170723.nav'
    # # PATH_JZ = r'/media/fruitspec-lab/cam175/USXXXX_citrus_lihi_temp/MAZMANI2/170723/row_20/1/jaized_timestamps.csv'
    # # PATH_TRANSLATIONS = "/media/fruitspec-lab/cam175/USXXXX_citrus_lihi_temp/MAZMANI2/170723/row_20/1/jai_translation.csv"
    #
    # row_path = r'/home/lihi/FruitSpec/Data/grapes/USXXXX/GRAPES/JACFAM/204401XX/180723/row_3/1'
    # PATH_GPS =  r'/home/lihi/FruitSpec/Data/grapes/USXXXX/GRAPES/JACFAM/204401XX/180723.nav'
    #
    # PATH_JZ = os.path.join(row_path,'jaized_timestamps.csv')
    # #PATH_TRANSLATIONS = os.path.join(row_path,'jai_translations.csv')
    # # PATH_JZ = r'/home/lihi/FruitSpec/code/lihi/fsCounter/vision/trees_slicer/slice_by_distance_using_tx_translations/data/roi_row_debug/jaized_timestamps.csv'
    # # PATH_TRANSLATIONS = "/home/lihi/FruitSpec/code/lihi/fsCounter/vision/trees_slicer/slice_by_distance_using_tx_translations/data/roi_row_debug/jai_translations.csv"
    #
    # path_slices_file = f"{row_path}/slices.csv"
    #
    # out_df, df = slice_frames(PATH_JZ, PATH_GPS, PATH_OUTPUT = path_slices_file , split_range = 3)
    # # save df:
    # path_df_file = f"{row_path}/gps_distance.csv"
    # df.to_csv(path_df_file)
    # #updated_df = slice_inside_frames(path_slices_csv=path_slices_file, path_translations_csv=PATH_TRANSLATIONS, frame_width=1535, output_path=f"{row_path}/all_slices.csv")
    # print('Done')
