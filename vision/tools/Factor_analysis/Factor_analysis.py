import os
def get_repo_dir():
    cwd = os.getcwd()
    splited = cwd.split('/')
    ind = splited.index('fsCounter')
    repo_dir = '/'
    for s in splited[1:ind + 1]:
        repo_dir = os.path.join(repo_dir, s)

    return repo_dir

import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sys.path.append(get_repo_dir())

from vision.tools.jupyter_notebooks.notebook_analysis_help_funcs import *
from vision.tools.post_process_analysis import read_tracks_and_slices, get_block_count
from vision.visualization.draw_bb_from_csv import draw_tree_bb_from_tracks


def concat_to_meta(block_meta, df):
    df_col = list(block_meta.columns)
    df['block'] = df['block'].str.lower()
    new_data = []
    data = block_meta.copy()
    for id_, sample in data.iterrows():
        block = sample['block'].lower()
        row = sample['row'].lower()
        tree_id = int(sample['tree_id'])
        # print(f'block == "{block}" and row == "{row}" and tree_id == {tree_id}')
        q_data = df.query(f'block == "{block}" and row == "{row}" and tree_id == {tree_id}')
        new_sample = sample.to_list()
        for i in range(1, 4):
            new_sample.append(q_data[str(i)].values[0])
        new_data.append(new_sample)

    df_col += ['cv1', 'cv2', 'cv3']
    new_df = pd.DataFrame(new_data, columns=df_col)

    return new_df


def add_ratios(df):
    df['F/cv1'] = df['F'] / df['cv1']
    df['F/cv2'] = df['F'] / df['cv2']
    df['F/cv3'] = df['F'] / df['cv3']

    return df


def get_block_ratio(block_df, row_tracks, y_threshold=800, depth=3):
    block_col = list(block_df.columns)
    new_data = []
    for id_, sample in block_df.iterrows():
        row = sample['row'].lower()
        tree_id = int(sample['tree_id'])
        tree_df = row_tracks[row][tree_id]

        d_tree_df = tree_df.query(f'depth <= {depth}')

        lower_tree_df = tree_df.query(f'y1 > {y_threshold} and depth <= {depth}')
        count = len(tree_df.track_id.unique())
        lower_count = len(lower_tree_df.track_id.unique())
        ratio = lower_count / count

        gdf = lower_tree_df.groupby('track_id')
        lower_tracks_depth = np.array(gdf.depth.mean())
        filtered_lower_tracks_depth = lower_tracks_depth[lower_tracks_depth < 3]
        mean = np.mean(filtered_lower_tracks_depth)
        std = np.std(filtered_lower_tracks_depth)

        new_sample = sample.to_list()

        uniq, counts = np.unique(d_tree_df["track_id"], return_counts=True)
        for i in range(0, 5):
            new_sample.append(len(uniq[counts > i]))

        new_data.append(new_sample)

    # block_col += ['y_ratio', 'mean', 'std', 'lcv1', 'lcv2', 'lcv3', 'lcv4', 'lcv5']
    block_col += ['dcv1', 'dcv2', 'dcv3', 'dcv4', 'dcv5']
    new_df = pd.DataFrame(new_data, columns=block_col)

    return new_df


def linear_model_selection(data, selection_cols=["cv1"], type_col="block", cross_val='row'):
    factors = {}
    for col in selection_cols:
        factor, res_mean, res_std, tree_mean, tree_std, all_preds = run_LROCV(data, cv_col=col, type_col=type_col,
                                                                              cross_val=cross_val, return_res=True)
        factors[col] = {'factor': factor, 'mean_error': res_mean, 'std_error': res_std}

    return factors


def block_analysis(block_path, metadata_path, block_):
    block_counts, row_tracks = get_block_count(block_path)
    block_counts_df = pd.DataFrame(block_counts, columns=['tree_id', 'block', 'row', '1', '2','3'])
    meta_data = pd.read_csv(metadata_path)
    block_meta = meta_data.query(f'block == "{block_}"')
    block_df = concat_to_meta(block_meta, block_counts_df)
    block_df['F/cv1'] = block_df['F'] / block_df['cv1']
    block_df= add_ratios(block_df)
    block_df = get_block_ratio(block_df, row_tracks)

    return block_df, row_tracks

def get_selection_error(factors_dict, block_df):
    results = {}
    for item_ in list(factors_dict.keys()):
        block_df[f'err_{item_}'] = (block_df['F'] - (block_df[item_] * factors_dict[item_]['factor'])) / block_df['F']
        results[item_] = {'err': np.mean(block_df[f'err_{item_}']), 'err_std': np.std(block_df[f'err_{item_}'])}

    return results, block_df

###########################

if __name__ == "__main__":

    OUTPUT_PATH = r'/home/fruitspec-lab-3/FruitSpec/Data/grapes/SAXXXX/Factors_analysis/conf04_nms005_windows1_2-1_5_cvd3_0047'

    metadata_path = "/home/fruitspec-lab-3/FruitSpec/Data/grapes/SAXXXX/counting/data_meta.csv"

    blocks_list = ['/home/fruitspec-lab-3/FruitSpec/Data/grapes/SAXXXX/1XXXXXX4',
                   '/home/fruitspec-lab-3/FruitSpec/Data/grapes/SAXXXX/5XXXXXX2',
                   '/home/fruitspec-lab-3/FruitSpec/Data/grapes/SAXXXX/7XXXXXX2',
                   '/home/fruitspec-lab-3/FruitSpec/Data/grapes/SAXXXX/8XXXXXX3']

    blocks_df = pd.DataFrame()
    res_concatenated_df = pd.DataFrame()
    for block_path in blocks_list:

        #block_path = "/home/fruitspec-lab-3/FruitSpec/Data/grapes/SAXXXX/5XXXXXX2"
        block_ = block_path.split('/')[-1]
        costumer = block_path.split('/')[-2]
        block_df, row_tracks = block_analysis(block_path, metadata_path, block_)

        # for index, row in block_df.iterrows():
        #     row_to_drow = row['row']
        #     tree_id = row['tree_id']
        #     date = '281123'
        #     tree_tracks = row_tracks[row_to_drow][tree_id]
        #     #######################
        #     #remove tracks in depth
        #     tree_tracks = tree_tracks[tree_tracks['depth'] <= 3] # todo added depth fillering
        #     ######################
        #     draw_tree_bb_from_tracks(tree_tracks, os.path.join(block_path, date, row_to_drow, '1'),
        #                              tree_id)
        #
        #     print ('ok')

        # row_to_drow = 'row_5'
        # tree_id = 1
        # date = '281123'
        # draw_tree_bb_from_tracks(row_tracks[row_to_drow][tree_id], os.path.join(block_path, date, row_to_drow, '1'), tree_id)
        #
        #
        # plt.bar(block_df['row'], block_df['F/cv1'])
        # plt.xlabel('Tree')
        # plt.ylabel('F/CV1')
        # plt.show()
        #
        #plot_F_cv(block_df, 1, add_xy_line=False)


        factors_dict = linear_model_selection(block_df, selection_cols=['cv1', 'dcv1', 'cv2', 'dcv2', 'cv3', 'dcv3'], type_col="block", cross_val='row')

        res, block_df = get_selection_error(factors_dict, block_df)
        res_df = pd.DataFrame.from_dict(res, orient='index').reset_index()
        res_df.rename(columns={'index': 'variable'}, inplace=True)
        res_df.insert(0, 'block', block_)
        res_df['factor'] = res_df['variable'].apply(
            lambda x: factors_dict[x]['factor'][0] if x in factors_dict else None)

        blocks_df = pd.concat([blocks_df, block_df.copy()], ignore_index=True)
        res_concatenated_df = pd.concat([res_concatenated_df, res_df.copy()], ignore_index=True)

    blocks_df.to_csv(os.path.join(OUTPUT_PATH, 'all_blocks.csv'))

    blocks_df.insert(0, 'costumer', costumer)
    # Remove block 7:
    blocks_df = blocks_df[blocks_df['block'] != '7XXXXXX2']
    factors_combined_dict = linear_model_selection(blocks_df, selection_cols=['cv1', 'dcv1', 'cv2', 'dcv2', 'cv3', 'dcv3'],
                                          type_col="costumer", cross_val="block")
    res_all, blocks_df = get_selection_error(factors_combined_dict, blocks_df)
    res_all_df = pd.DataFrame.from_dict(res_all, orient='index').reset_index()
    res_all_df.rename(columns={'index': 'variable'}, inplace=True)
    res_all_df['factor'] = res_all_df['variable'].apply(
        lambda x: factors_combined_dict[x]['factor'][0] if x in factors_combined_dict else None)

    res_all_df.to_csv(os.path.join(OUTPUT_PATH, 'results_all_blocks_but_7.csv'))
    res_concatenated_df.to_csv(os.path.join(OUTPUT_PATH, 'results_each_block.csv'))
    blocks_df.to_csv(os.path.join(OUTPUT_PATH, 'all_blocks.csv'))



    print ('ok')
