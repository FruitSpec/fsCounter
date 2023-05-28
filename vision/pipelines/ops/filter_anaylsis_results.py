import os
import numpy as np
import pandas as pd


def filter_results(data, columns, count_threshold, depth_threshold):

    unique_track_ids, counts = get_track_ids_and_count(data, columns)
    tracks_to_filter = ids_to_filter_by_threshold(counts, count_threshold)





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


def ids_to_filter_by_threshold(values, threshold):
    to_filter = np.argwhere(values < threshold)

    return to_filter








