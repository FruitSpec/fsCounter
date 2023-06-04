import os
from math import radians, cos, sin, asin, sqrt
import numpy as np
import pandas as pd


def get_validated_tracks(data, columns, count_threshold, depth_threshold):
    data = filter_results(data, columns, count_threshold, depth_threshold)
    tracks_col = get_column_number(columns)
    number_of_validated_tracks = len(np.unique(data[:, tracks_col]))

    return number_of_validated_tracks

def filter_results(data, columns, count_threshold, depth_threshold):

    data = get_data_filtered_by_count(data, columns, count_threshold)
    data = get_data_filtered_by_depth(data, columns, depth_threshold)

    return data
def get_data_filtered_by_count(data, columns, count_threshold):
    unique_track_ids, counts = get_track_ids_and_count(data, columns)
    ids_to_keep = np.argwhere(counts >= count_threshold)
    unique_track_ids = unique_track_ids[ids_to_keep]
    ids = unique_to_data_ids(data, columns, unique_track_ids)

    return data[ids]

def get_data_filtered_by_depth(data, columns, depth_threshold):
    col_id = get_column_number(columns, 'depth')
    ids_to_keep = np.argwhere(data[:, col_id] < depth_threshold)
    data = data[ids_to_keep.flatten()]

    return data

def load_csv(csv_path):
    data = pd.read_csv(csv_path)
    columns = list(data.columns)

    return data.to_numpy(), columns

def get_column_number(columns, col_name='track_id'):

    found = False
    for i, v in enumerate(columns):
        if v == col_name:
            found = True
            break

    if not found:
        print(f"{col_name} is not in file header")
        raise KeyError

    return i

def get_track_ids_and_count(data, columns):

    column_id = get_column_number(columns)
    unique_track_ids, counts = np.unique(data[:, column_id], return_counts=True)

    return unique_track_ids, counts

def unique_to_data_ids(data, columns, unique_track_ids):
    column_id = get_column_number(columns)
    track_ids = data[:, column_id]
    ids_to_keep = []
    for id_, trk in enumerate(track_ids):
        if trk in unique_track_ids:
            ids_to_keep.append(id_)

    return ids_to_keep


def distance(lat1, lat2, lon1, lon2):
    # convert from degrees to radians
    lon1 = radians(lon1)
    lon2 = radians(lon2)
    lat1 = radians(lat1)
    lat2 = radians(lat2)

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2

    c = 2 * asin(sqrt(a))

    # Radius of earth in kilometers. Use 3956 for miles
    r = 6371

    # calculate the result
    return (c * r) * 1000  # results in M



if __name__ == "__main__":

    fp = "/home/matans/Documents/fruitspec/sandbox/Apples_Golan_heights/MED00000/230523/row_1/1/tracks.csv"
    depth_threshold = 5#3.5
    count_threshold = 3
    data, columns = load_csv(fp)
    t = get_validated_tracks(data, columns, count_threshold, depth_threshold)
    print(t)