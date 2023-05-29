import os
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
    data = data[ids_to_keep.flatten()]

    return data

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



if __name__ == "__main__":

    fp = "/home/matans/Documents/fruitspec/sandbox/Apples_Golan_heights/OLD00000/230523/row_1/tracks_1.csv"
    depth_threshold = 3.5
    count_threshold = 3
    data, columns = load_csv(fp)
    t = get_validated_tracks(data, columns, count_threshold, depth_threshold)