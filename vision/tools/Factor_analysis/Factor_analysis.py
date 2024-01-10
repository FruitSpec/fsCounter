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
from vision.misc.help_func import validate_output_path
from vision.tools.utils_general import file_exists


def concat_to_meta(block_meta, df):
    df_col = list(block_meta.columns)
    df['block'] = df['block'].str.lower()
    new_data = []
    data = block_meta.copy()
    for id_, sample in data.iterrows():
        block = sample['block'].lower()
        row = sample['row'].lower()
        tree_id = int(sample['tree_id'])
        print(f'block == "{block}" and row == "{row}" and tree_id == {tree_id}')
        q_data = df.query(f'block == "{block}" and row == "{row}" and tree_id == {tree_id}')
        new_sample = sample.to_list()
        for i in range(1, 4):
            if len(q_data) == 0:
                new_sample.append(-1)
            else:
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
        if not tree_id in list(row_tracks[row].keys()):
            continue
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


def linear_model_selection(data, selection_cols=["cv1"], type_col="block", cross_val='row', draw_plot = False):
    factors = {}
    for col in selection_cols:
        print (col)
        factor, res_mean, res_std, tree_mean, tree_std, all_preds = run_LROCV(data, cv_col=col, type_col=type_col,
                                                                              cross_val=cross_val, return_res=True)
        factors[col] = {'factor': factor, 'mean_error': res_mean, 'std_error': res_std}
        if draw_plot:
            create_scatter_plot_with_hue_and_line(data, all_preds, title =f'{col}_at_factor_{round(factor[0], 3)}')
    return factors


def create_scatter_plot_with_hue_and_line(data, all_preds, title, factor, axes):
    unique_blocks = data['block'].unique()
    for block in unique_blocks:
        axes.scatter(all_preds[data['block'] == block], data['F'][data['block'] == block], label=f'{block}')

    x = np.linspace(0, data['F'].max(), len(all_preds))
    y = 1 * x  # slope
    #axes.plot(x, y, label='perfect pred', color='gray')
    axes.plot(x, y * factor, label=f'regression line {round(factor,2)}', color='green')

    max_value = data['F'].max()
    axes.set_xlim(0, max_value + 10)
    axes.set_ylim(0, max_value + 10)

    axes.set_xlabel('CV')
    axes.set_ylabel('Fruit')
    axes.set_title(title)
    axes.legend()


def block_analysis(block_path, metadata_path, block_, depth=3):
    block_counts, row_tracks = get_block_count(block_path)
    block_counts_df = pd.DataFrame(block_counts, columns=['tree_id', 'block', 'row', '1', '2','3'])
    meta_data = pd.read_csv(metadata_path)
    block_meta = meta_data.query(f'block == "{block_}"')
    block_df = concat_to_meta(block_meta, block_counts_df)
    block_df['F/cv1'] = block_df['F'] / block_df['cv1']
    block_df = add_ratios(block_df)
    block_df = get_block_ratio(block_df, row_tracks, depth=depth)

    return block_df, row_tracks

def get_selection_error(factors_dict, block_df):
    results = {}
    for item_ in list(factors_dict.keys()):
        block_df[f'err_{item_}'] = (block_df['F'] - (block_df[item_] * factors_dict[item_]['factor'])) / block_df['F']
        results[item_] = {'err': np.mean(block_df[f'err_{item_}']), 'err_std': np.std(block_df[f'err_{item_}'])}

    return results, block_df

def scatter_plot(blocks_df, cvs = ['cv3','dcv3'], title = 'before_factor'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  # Adjust figsize as needed
    create_scatter_plot_with_hue_and_line(blocks_df, blocks_df[cvs[0]], f"Fruit_vs_{cvs[0]}_{title}", factor = blocks_df[f'factor_{cvs[0]}'][0], axes = ax1)
    create_scatter_plot_with_hue_and_line(blocks_df, blocks_df[cvs[1]], f"Fruit_vs_{cvs[1]}_{title}", factor = blocks_df[f'factor_{cvs[1]}'][0], axes = ax2)
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_PATH, f"{title}.png"))
    print(f'Saved: {os.path.join(OUTPUT_PATH, f"{title}.png")}')


def draw_tree_bb_for_block(block_path, block_df, row_tracks, dir='1'):

    for index, row in block_df.iterrows():
        row_to_drow = row['row']
        tree_id = row['tree_id']
        dates = os.listdir(block_path)
        dates = [item for item in dates if item.isdigit()]

        for date in dates:
            tree_tracks = row_tracks[row_to_drow][tree_id]
            tree_tracks['track_id'] = np.where(tree_tracks['depth'] <= 3, 1, 0)  # To color depth

            draw_tree_bb_from_tracks(tree_tracks, os.path.join(block_path, date, row_to_drow, dir),
                                     tree_id, data_index=6)


def plot_errors_and_abs_errors(df, output_path):
    # Grouping the data by 'block' and calculating the mean of errors for each block
    grouped_df = df.groupby('block')[['err_cv3', 'err_dcv3']].mean().reset_index()

    # Calculate absolute errors
    grouped_df['abs_err_cv3'] = grouped_df['err_cv3'].abs()
    grouped_df['abs_err_dcv3'] = grouped_df['err_dcv3'].abs()

    # Setting up the plot positions and width for the bars
    pos = list(range(len(grouped_df['block'])))
    width = 0.35

    # Creating two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

    # Subplot 1: Regular Error
    ax1.bar([p - width / 2 for p in pos], grouped_df['err_cv3'], width, alpha=0.5, label='err_cv3')
    ax1.bar([p + width / 2 for p in pos], grouped_df['err_dcv3'], width, alpha=0.5, label='err_dcv3')
    ax1.set_ylabel('Error')
    ax1.set_title('Error with factor')
    ax1.set_xticks(pos)
    ax1.set_xticklabels(grouped_df['block'])
    ax1.legend(['err_cv3', 'err_dcv3'], loc='upper right')
    ax1.grid()

    # Subplot 2: Absolute Error
    ax2.bar([p - width / 2 for p in pos], grouped_df['abs_err_cv3'], width, alpha=0.5, label='abs_err_cv3')
    ax2.bar([p + width / 2 for p in pos], grouped_df['abs_err_dcv3'], width, alpha=0.5, label='abs_err_dcv3')
    ax2.set_ylabel('Absolute Error')
    ax2.set_title('Absolute Error with factor')
    ax2.set_xticks(pos)
    ax2.set_xticklabels(grouped_df['block'])
    ax2.legend(['abs_err_cv3', 'abs_err_dcv3'], loc='upper right')
    ax2.grid()

    plt.tight_layout()
    fig.savefig(f"{output_path}/errors_plot.png")
    print (f'Saved: {f"{output_path}/errors_plot.png"}')


def process_blocks_means_results(blocks_df, output_path):
    """
    Process block data by calculating mean values, standard deviations,
    adding a row for the mean of absolute values, converting specific columns to integers,
    and rounding other columns.
    """

    # Drop columns not needed for calculating the mean
    cols_to_drop = ['id', 'row', 'tree', 'side', 'tree_id',
                    'pred_cv1', 'pred_dcv1', 'pred_cv2', 'pred_dcv2', 'pred_cv3', 'pred_dcv3']
    reduced_df = blocks_df.drop(cols_to_drop, axis=1)

    # Calculate mean values per block and rename columns
    mean_values_per_block_df = reduced_df.groupby('block').mean().reset_index()
    mean_values_per_block_df = mean_values_per_block_df.rename(
        columns={col: f'{col}_mean' if col != 'block' else col for col in mean_values_per_block_df})

    # Calculate standard deviation for error columns and rename
    existing_error_columns = [col for col in blocks_df.columns if col.startswith('error_')]
    std_error_per_block_df = blocks_df.groupby('block')[existing_error_columns].std().reset_index()
    std_error_per_block_df = std_error_per_block_df.rename(
        columns={col: f'std_{col}' for col in existing_error_columns})

    # Merge mean values and standard deviation data
    mean_values_per_block_df = pd.merge(mean_values_per_block_df, std_error_per_block_df, on='block')

    # Calculate and add a row for the mean of absolute values
    abs_mean = mean_values_per_block_df.drop('block', axis=1).abs().mean()
    new_row = pd.DataFrame([['abs_mean'] + abs_mean.tolist()], columns=mean_values_per_block_df.columns)
    mean_values_per_block_df = mean_values_per_block_df.append(new_row, ignore_index=True)

    # Convert specific columns to integers
    int_columns = ['F_mean', 'cv1_mean', 'cv2_mean', 'cv3_mean', 'dcv1_mean', 'dcv2_mean', 'dcv3_mean', 'dcv4_mean', 'dcv5_mean']
    mean_values_per_block_df[int_columns] = mean_values_per_block_df[int_columns].astype(int)

    # Round other columns to 4 decimal places
    other_columns = [col for col in mean_values_per_block_df.columns if col not in int_columns]
    mean_values_per_block_df[other_columns] = mean_values_per_block_df[other_columns].round(3)

    # save:
    mean_values_per_block_df.to_csv(output_path)
    print (f"Saved: {output_path}")

    return mean_values_per_block_df

def add_factors(blocks_df, factors_combined_dict, cvs):

    # Adding factor values to blocks_df:
    for cv in cvs:
        blocks_df[f'factor_{cv}'] = factors_combined_dict[cv]['factor'][0]
    return blocks_df

def add_preds_errors_sdts(blocks_df, cvs, output_path):

    # Adding prediction columns by multiplying corresponding cv values with their factors
    for cv_dcv in cvs:
        blocks_df[f'pred_{cv_dcv}'] = blocks_df[cv_dcv] * blocks_df[f'factor_{cv_dcv}']

    # Calculating percentage errors for each prediction
    for cv_dcv in cvs:
        blocks_df[f'error_{cv_dcv}'] = (blocks_df['F'] - blocks_df[f'pred_{cv_dcv}']) / blocks_df['F']

    # blocks_df.to_csv(os.path.join(OUTPUT_PATH, 'Factor_Analysis_multi_factors.csv'))
    blocks_df.to_csv(output_path)
    print (f"Saved: {output_path}")

    return blocks_df

def factor_analysis(METADATA_PATH, BLOCKS_LIST, OUTPUT_PATH, DEPTH_FILTER, CVS, DRAW_TREES = False):
    blocks_df = pd.DataFrame()
    blocks_multi_factors_df = pd.DataFrame()

    for block_path in BLOCKS_LIST:

        block_ = block_path.split('/')[-1]
        print(f'***  Block {block_}  ***')
        customer = block_path.split('/')[-2]
        block_df, row_tracks = block_analysis(block_path, METADATA_PATH, block_, DEPTH_FILTER)
        blocks_df = pd.concat([blocks_df, block_df.copy()], ignore_index=True)
        if DRAW_TREES:
            draw_tree_bb_for_block(block_path, block_df, row_tracks, dir='1')

        # plt.bar(block_df['row'], block_df['F/cv1'])
        # plt.xlabel('Tree')
        # plt.ylabel('F/CV1')
        # plt.show()

        factors_dict = linear_model_selection(block_df, selection_cols = CVS, type_col="block", cross_val='row')
        block_multi_factors_df = add_factors(block_df.copy(), factors_dict, CVS)
        blocks_multi_factors_df = pd.concat([blocks_multi_factors_df, block_multi_factors_df.copy()], ignore_index=True)




    blocks_multi_factors_df = add_preds_errors_sdts(blocks_multi_factors_df, CVS, output_path = os.path.join(OUTPUT_PATH, 'Factor_Analysis_multi_factors.csv'))
    multi_factors_means_results = process_blocks_means_results(blocks_multi_factors_df, output_path = os.path.join(OUTPUT_PATH, 'Factor_Analysis_multi_factors_blocks_means.csv'))

    # Get factors and analysis with one factor for all blocks:
    blocks_df.insert(0, 'customer', customer)

    # Linear regression:
    factors_combined_dict = linear_model_selection(blocks_df, selection_cols= CVS, type_col="customer", cross_val="block")

    blocks_df = add_factors(blocks_df, factors_combined_dict, CVS)
    blocks_df = add_preds_errors_sdts(blocks_df, CVS, output_path = os.path.join(OUTPUT_PATH, 'Factor_Analysis_single_factor.csv'))
    df_blocks_means_results = process_blocks_means_results(blocks_df, output_path = os.path.join(OUTPUT_PATH, 'Factor_Analysis_single_factor_blocks_means.csv'))

    # plot all blocks before factor:
    scatter_plot(blocks_df, cvs = ['cv3','dcv3'], title='before_factor')
###########################

if __name__ == "__main__":

    METADATA_PATH = "/home/lihi/FruitSpec/Data/SA/CITRUS/CAPESPN/Data_files/data_meta_2024-01-10_11-43-49.csv"

    # BLOCKS_LIST = ['/home/fruitspec-lab-3/FruitSpec/Data/grapes/SAXXXX/1XXXXXX4',
    #                '/home/fruitspec-lab-3/FruitSpec/Data/grapes/SAXXXX/3XXXXXX4',
    #                '/home/fruitspec-lab-3/FruitSpec/Data/grapes/SAXXXX/5XXXXXX2',
    #                '/home/fruitspec-lab-3/FruitSpec/Data/grapes/SAXXXX/8XXXXXX3',
    #                '/home/fruitspec-lab-3/FruitSpec/Data/grapes/SAXXXX/9XXXXXX3',
    #                '/home/fruitspec-lab-3/FruitSpec/Data/grapes/SAXXXX/14XXXXX2']

    # Get the full path of all blocks in the directory:
    root_path = f'/home/lihi/FruitSpec/Data/SA/CITRUS/CAPESPN'
    BLOCKS_LIST = []
    for entry in os.scandir(root_path):
        if entry.is_dir() and entry.name[0].isdigit():
            # Construct the full path and add it to the list
            full_path = entry.path
            BLOCKS_LIST.append(full_path)

    OUTPUT_PATH = os.path.join(root_path, 'Factors_analysis')
    DEPTH_FILTER = 3
    CVS = ['cv1', 'dcv1', 'cv2', 'dcv2', 'cv3', 'dcv3']
    DRAW_TREES = False

##############################################
    for block_path in BLOCKS_LIST:
        block = block_path.split('/')[-1]
        block_dates = os.listdir(block_path)
        block_dates = [item for item in block_dates if item.isdigit()]

        for date in block_dates:
            date_path = os.path.join(block_path, date)
            if not os.path.isdir(date_path):
                continue
            row_list = os.listdir(date_path)

            for row in row_list:

                row_path = os.path.join(date_path, row)
                if not os.path.isdir(row_path):
                    continue
                row_path = os.path.join(row_path, '1')
                if not os.path.exists(row_path):
                    continue

                tracks_path = os.path.join(row_path, 'tracks.csv')
                ex = file_exists(tracks_path, raise_error=False)
                print (f'{row_path} - TRECKS EXIST: ({ex})')


 ###########################################
    factor_analysis(METADATA_PATH, BLOCKS_LIST, OUTPUT_PATH, DEPTH_FILTER, CVS, DRAW_TREES)

    print ('Done')
