import numpy as np
import pandas as pd


def preprocess(df):
    df["fruit_foliage_ratio"] = df["fruit_foliage_ratio"].replace(np.inf, 100)
    df_cols = df.columns
    for col in ["F", "block_name", "side", "tree_number", "name"]:
        if col in df_cols:
            df.drop(col, axis=1, inplace=True)
    df['cv^2'] = df["cv"]**2
    df['cv/center_height'] = df["cv"]/df["center_height"]
    df['cv/center_width'] = df["cv"]/df["center_width"]
    df['cv/surface_area'] = df["cv"]/df["surface_area"]
    df['cv/total_foliage'] = df["cv"]/df["total_foliage"]
    df['cv*fruit_foliage_ratio'] = df["cv"]*df["fruit_foliage_ratio"]
    df['mst_sums_arr^2'] = df["mst_sums_arr"]**2
    return df
