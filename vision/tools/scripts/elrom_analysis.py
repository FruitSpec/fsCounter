import os
import numpy as np
import pandas as pd

from vision.pipelines.ops.filter_anaylsis_results import get_validated_tracks, load_csv

def run_on_rows(rows_path, tracks_threshold, depth_threshold):

    res = []
    rows_list = os.listdir(rows_path)
    for row in rows_list:
        row_path = os.path.join(rows_path, row)
        row_tests_list = os.listdir(row_path)
        for test in row_tests_list:
            if 'alignment' in test:
                continue

            data, columns = load_csv(os.path.join(row_path, test))
            count = get_validated_tracks(data, columns, tracks_threshold, depth_threshold)

            name = test.split('.')[0]
            scan_number = int(name.split('_')[-1])

            scan_id = 2 if scan_number % 2 == 0 else 1

            res.append({'row': row, 'side': scan_id, 'count': count, 'id': scan_number})

    df = pd.DataFrame(res, columns=['row', 'side', 'count', 'id'])
    return df


if __name__ == "__main__":
    rows_path = "/home/matans/Documents/fruitspec/sandbox/Apples_Golan_heights/ELROM_test/Repeatability"
    tracks_threshold = 2
    depth_threshold = 5
    df = run_on_rows(rows_path, tracks_threshold, depth_threshold)
    df.to_csv("/home/matans/Documents/fruitspec/sandbox/Apples_Golan_heights/ELROM_test/res_2.csv")