import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from vision.tools.utils_general import find_subdirs_with_file


def process_data(df, cv, group_by):
    stats = df.groupby(['block', group_by])[cv].agg(['mean', 'std']).reset_index()
    stats[group_by] = pd.Categorical(stats[group_by])
    return stats.sort_values(by=['block', group_by])

def plot_data(df, variables, group_by, file_path):
    fig, axes = plt.subplots(nrows=len(variables), ncols=1, figsize=(15, 24))

    for i, var in enumerate(variables):
        stats_sorted = process_data(df, var, group_by)
        stats_sorted.to_csv(f'/home/fruitspec-lab-3/FruitSpec/Data/customers/Israel/Data_files_old_FSI/{group_by}_{var}_summary.csv')

        # Plotting
        bar_plot = sns.barplot(x='block', y='mean', hue=group_by, data=stats_sorted, palette="Set2", ax=axes[i])

        # Error bars
        n_categories = len(stats_sorted[group_by].unique())
        total_width = 0.8  # Total width that seaborn uses for each group
        offset = total_width / (2 * n_categories)  # Offset to align error bars

        for idx, row in stats_sorted.iterrows():
            block_idx = list(stats_sorted['block'].unique()).index(row['block'])
            cat_idx = list(stats_sorted[group_by].unique()).index(row[group_by])
            position = block_idx + (total_width / n_categories) * (cat_idx - n_categories / 2) + offset
            axes[i].errorbar(position, row['mean'], yerr=row['std'], fmt='none', ecolor='black', capsize=5, elinewidth=2)

        # Titles and labels
        axes[i].set_title(f'{var.upper()} by {group_by}')
        axes[i].set_xlabel('Block')
        axes[i].set_ylabel(f'Average {var.upper()} Value')
        axes[i].legend(title=group_by)

    plt.tight_layout()
    plt.savefig(file_path)
    plt.show()

if __name__ == "__main__":

    folder_path = '/home/fruitspec-lab-3/FruitSpec/Data/customers/Israel'
    file_name = 'tracks.csv'

    metadata_path = f'/home/fruitspec-lab-3/FruitSpec/Data/customers/Israel/Data_files/exp_meta.csv'
    df_metadata = pd.read_csv(metadata_path)
    df_metadata['cv1'] = None
    df_metadata['cv2'] = None
    df_metadata['cv3'] = None
    tracks_paths = find_subdirs_with_file(folder_path, file_name, return_dirs=False, single_file=False)

    for tracks_path in tracks_paths:
        block, row = tracks_path.split('/')[-5], tracks_path.split('/')[-3].split('_')[-1]
        df_tracks = pd.read_csv(tracks_path)

        for min_samples, cv in zip ([1,2,3], ['cv1', 'cv2', 'cv3']):
            unique_tracks, counts = np.unique(df_tracks["track_id"], return_counts=True)
            n = len(unique_tracks[counts >= min_samples])

            # Insert the value
            condition = (df_metadata['block'] == block) & (df_metadata['row'] == int(row))
            df_metadata.loc[condition, cv] = n


    df_metadata.to_csv(r'/home/fruitspec-lab-3/FruitSpec/Data/customers/Israel/Data_files/exp_meta.csv')

    #################################################################################################################

    # Load the data
    df = pd.read_csv(r'/home/fruitspec-lab-3/FruitSpec/Data/customers/Israel/Data_files_old_FSI/exp_meta.csv')

    # Define the variables and grouping column
    cvs = ['cv1', 'cv2', 'cv3']

    group_by_sun_direction = 'Sun_direction'
    plot_data(df, cvs, group_by_sun_direction, '/home/fruitspec-lab-3/FruitSpec/Data/customers/Israel/Data_files/avg_plot_by_sun_direction1.png')


    # Plotting for day_time
    group_by_day_time = 'day_time'
    plot_data(df, cvs, group_by_day_time, '/home/fruitspec-lab-3/FruitSpec/Data/customers/Israel/Data_files/avg_plot_by_daytime.png')