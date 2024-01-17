import os
import sys

import pandas as pd
from tqdm import tqdm

from sklearn.cluster import KMeans

from vision.misc.help_func import get_repo_dir, validate_output_path

sys.path.append(get_repo_dir())

from vision.tools.jupyter_notebooks.notebook_analysis_help_funcs import *
from vision.tools.post_process_analysis import read_tracks_and_slices, get_block_count, count_trees_fruits, coarse_filter_depth, fine_filter_depth
from vision.visualization.draw_bb_from_csv import draw_tree_bb_from_tracks


def analyze_section(section_df, min_samp=3, dist_center_threshold=150, color_break_threshold=0.75, debug=None):

    tracks_updated, picked_clusters, tracks_to_color = get_picked_in_section(section_df,
                                                                             min_samp,
                                                                             dist_center_threshold,
                                                                             color_break_threshold)
    tracks_width, tracks_height = get_tracks_width_and_height(tracks_updated)

    color_bins = get_color_bins(tracks_to_color)

    if debug is not None:
        draw_tree_bb_from_tracks(tracks_updated, debug['path'], debug['tree_id'], is_zed=True,
                                 data_index=debug['picked_col'], output_folder=debug['tree_output'])
        draw_tree_bb_from_tracks(tracks_updated, debug['path'], debug['tree_id'], is_zed=True,
                                 data_index=debug['cluster_col'], output_folder=debug['cluster_output'])

    return picked_clusters, color_bins, tracks_to_color,  tracks_width, tracks_height




def get_picked_in_section(section_df, min_samp=3, dist_center_threshold=150, color_break_threshold=0.75):
    uniq, counts = np.unique(section_df["track_id"], return_counts=True)

    """ filter tracks below threshold"""
    valid_tracks = uniq[counts >= min_samp]
    tracks_cleaned = section_df[np.isin(section_df["track_id"], valid_tracks)]

    """ filter depth"""
    tracks_updated = coarse_filter_depth(tracks_cleaned, 1.5)
    tracks_updated = fine_filter_depth(tracks_updated)

    """ curate cluster"""
    tracks_updated = curate_clusters(tracks_updated, dist_center_threshold)

    """ get picked clusters"""
    tracks_updated, tracks_to_color = argmax_tracks_colors(tracks_updated)
    tracks_updated, picked_clusters = is_picked(tracks_updated, tracks_to_color, color_break_threshold)

    tracks_to_color = filter_non_picked_tracks(tracks_updated, tracks_to_color)

    return tracks_updated, picked_clusters, tracks_to_color


def filter_non_picked_tracks(tracks_df, tracks_to_color):
    picked_df = tracks_df.query('is_picked == 1')
    picked_tracks = np.unique(picked_df['track_id'])

    final_tracks_to_color = dict()
    for key, value in tracks_to_color.items():
        if key in picked_tracks:
            final_tracks_to_color[key] = value

    return final_tracks_to_color



def get_clusters_average_depth(tracks_df):
    t_depth = tracks_df.groupby('track_id').depth.mean()
    c_ids = list(tracks_df['cluster_id'].unique())
    cluster_depth = {}
    for c in c_ids:
        c_df = tracks_df.query(f'cluster_id == {c}')
        c_t_ids = list(c_df['track_id'].unique())
        c_depths = []
        for t in c_t_ids:
            c_depths.append(t_depth[t])
        cluster_depth[c] = np.mean(c_depths)

    return cluster_depth


def curate_clusters(tracks_df, dist_center_threshold=150):
    tracks_to_clusters = get_tracks_to_cluster(tracks_df)

    tracks_df['center_x'] = (tracks_df['x1'] + tracks_df['x2']) / 2
    tracks_df['center_y'] = (tracks_df['y1'] + tracks_df['y2']) / 2


    for t_id, clusters in tracks_to_clusters.items():
        if len(clusters) == 2:
            first_cluster = clusters[0]
            second_cluster = clusters[1]
            
            if first_cluster == 292 or second_cluster == 292:
                a = 1

            first_df = tracks_df[tracks_df['cluster_id'] == first_cluster].copy()
            first_frames = first_df.frame.unique()

            second_df = tracks_df[tracks_df['cluster_id'] == second_cluster].copy()
            second_frames = second_df.frame.unique()

            if len(first_frames) == 0 or len(second_frames) == 0:
                continue

            mutual_frames = []
            for f_id in first_frames:
                if f_id in second_frames:
                    mutual_frames.append(f_id)
           
            if len(mutual_frames) == 0:  # no overlap
                # merge to first cluster
                tracks_df.loc[tracks_df['cluster_id'] == second_cluster, 'cluster_id'] = first_cluster  
           
            else:  # there is overlap.


                first_df_overlap = first_df[np.isin(first_df["frame"], mutual_frames)]
                second_df_overlap = second_df[np.isin(second_df["frame"], mutual_frames)]


                f_tracks = np.unique(first_df_overlap['track_id'])
                s_tracks = np.unique(second_df_overlap['track_id'])

                # at least one small cluster
                if len(f_tracks) <= 2 or len(s_tracks) <= 2:
                    dist = get_clusters_distance(first_df_overlap, second_df_overlap)
                    if dist <= dist_center_threshold:  # distance below threshold merge clusters
                        if len(f_tracks) <= len(s_tracks):
                            cluster_to_merge = first_cluster
                            cluster_to_update = second_cluster
                        else:
                            cluster_to_merge = second_cluster
                            cluster_to_update = first_cluster
                        tracks_df.loc[tracks_df['cluster_id'] == cluster_to_merge, 'cluster_id'] = cluster_to_update

                # too big - remove joined
                else:
                    stats = {}
                    for f_id in mutual_frames:
                        f_frame_df = first_df[first_df['frame'] == f_id]
                        s_frame_df = second_df[second_df['frame'] == f_id]


                        f_k_df = f_frame_df.loc[:,['center_x', 'center_y', 'depth']]
                        f_k_ids = f_frame_df.loc[:, 'track_id']
                        s_k_df = s_frame_df.loc[:, ['center_x', 'center_y', 'depth']]
                        s_k_ids = s_frame_df.loc[:, 'track_id']
                        k_ids = pd.concat([f_k_ids, s_k_ids], ignore_index=True).to_numpy()

                        f_k_center = f_k_df.mean()
                        f_k_center['center_x'] = f_k_center['center_x'] / 1080
                        f_k_center['center_y'] = f_k_center['center_y'] / 1920

                        s_k_center = s_k_df.mean()
                        s_k_center['center_x'] = s_k_center['center_x'] / 1080
                        s_k_center['center_y'] = s_k_center['center_y'] / 1920

                        init_arr = pd.concat([f_k_center, s_k_center], ignore_index=True).to_numpy()
                        init_arr = init_arr.reshape([2, 3])

                        combined_k_df = pd.concat([f_k_df, s_k_df], ignore_index=True)

                        combined_k_df['center_x'] = combined_k_df['center_x'] / 1080
                        combined_k_df['center_y'] = combined_k_df['center_y'] / 1920

                        kmeans = KMeans(n_clusters=2, random_state=0, max_iter=100, init=init_arr, n_init=1)
                        pred = kmeans.fit_predict(combined_k_df.to_numpy())

                        stats_ids = list(stats.keys())
                        for id_, t_id in enumerate(k_ids):
                            if t_id not in stats_ids:
                                stats[t_id] = {first_cluster: 0, second_cluster: 0}

                            if pred[id_] == 0:
                                stats[t_id][first_cluster] += 1
                            else:
                                stats[t_id][second_cluster] += 1


                    for t_id, counts in stats.items():
                        clusters_ids = list(counts.keys())
                        best_cluster = clusters_ids[0] if counts[clusters_ids[0]] > counts[clusters_ids[1]] else clusters_ids[1]
                        tracks_df.loc[tracks_df['track_id'] == t_id, 'cluster_id'] = int(best_cluster)

         
    return tracks_df

def get_track_dist_to_clusters(track_df, first_df, second_df, t_id):
    track_center_x = track_df.center_x.to_numpy()[0]
    track_center_y = track_df.center_y.to_numpy()[0]

    f_frame_tracks_exclusive = first_df[first_df['track_id'] != t_id]
    f_cluster_center_x = f_frame_tracks_exclusive.center_x.mean()
    f_cluster_center_y = f_frame_tracks_exclusive.center_y.mean()

    s_frame_tracks_exclusive = second_df[second_df['track_id'] != t_id]
    s_cluster_center_x = s_frame_tracks_exclusive.center_x.mean()
    s_cluster_center_y = s_frame_tracks_exclusive.center_y.mean()

    f_delta_x = np.abs(f_cluster_center_x - track_center_x)
    s_delta_x = np.abs(s_cluster_center_x - track_center_x)

    f_delta_y = np.abs(f_cluster_center_y - track_center_y)
    s_delta_y = np.abs(s_cluster_center_y - track_center_y)

    f_dist = np.sqrt(np.power(f_delta_x, 2) + np.power(f_delta_y, 2))
    s_dist = np.sqrt(np.power(s_delta_x, 2) + np.power(s_delta_y, 2))

    return f_dist, s_dist

def get_tracks_to_cluster(tracks_df):
    tracks_to_clusters = dict()
    for id_, track in tracks_df.iterrows():
        found_track_ids = list(tracks_to_clusters.keys())
        track_id = track['track_id']
        cluster_id = track['cluster_id']
        if track_id in found_track_ids:
            if cluster_id not in tracks_to_clusters[track_id]:
                tracks_to_clusters[track_id].append(cluster_id)
        else:
            tracks_to_clusters[track_id] = [cluster_id]
            
    return tracks_to_clusters

def get_clusters_distance(first_df, second_df):

    f_x_centers = first_df.groupby('frame').center_x.mean()
    f_y_centers = first_df.groupby('frame').center_y.mean()

    s_x_centers = second_df.groupby('frame').center_x.mean()
    s_y_centers = second_df.groupby('frame').center_y.mean()

    f_y = np.array(f_y_centers)
    f_x = np.array(f_x_centers)

    s_y = np.array(s_y_centers)
    s_x = np.array(s_x_centers)

    dist = np.mean(np.sqrt(np.power(f_y - s_y, 2) + np.power(f_x - s_x, 2)))

    return dist

def argmax_tracks_colors(tracks_df):
    track_ids = list(tracks_df['track_id'].unique())

    tracks_to_color = {}
    for track in track_ids:
        t_df = tracks_df.query(f'track_id == {track}')
        colors, count = np.unique(t_df['color'], return_counts=True)
        final_color_id = np.argmax(count)
        final_color = colors[final_color_id]
        tracks_df.loc[tracks_df['track_id'] == track, 'color'] = final_color
        if track not in list(tracks_to_color.keys()):
            tracks_to_color[track] = final_color

    return tracks_df, tracks_to_color


def is_picked(tracks_df, tracks_to_color, color_break_threshold=0.75):
    tracks_df['is_picked'] = 0
    picked_clusters = []
    breaking_colors = [1, 2, 3, 4]  # all other than green
    c_ids = list(tracks_df['cluster_id'].unique())
    for c in c_ids:
        cluster_picked = False
        cluster_colors = []
        c_df = tracks_df.query(f'cluster_id == {c}')

        track_ids = list(c_df['track_id'].unique())
        for track in track_ids:
            cluster_colors.append(int(tracks_to_color[track]))

        if len(cluster_colors) < 3:
            for color in cluster_colors:
                if color in breaking_colors:
                    cluster_picked = True
        else:
            color_breaking_ratio = np.sum(np.array(cluster_colors) < 5) / len(cluster_colors)
            if color_breaking_ratio >= color_break_threshold:
                cluster_picked = True

        if cluster_picked:
            tracks_df.loc[tracks_df['cluster_id'] == c, 'is_picked'] = 1
            picked_clusters.append(c)

    return tracks_df, picked_clusters


def get_tracks_width_and_height(tracks_df, upper_filter=0.09, lower_filter=0.04):
    """ default filters assume tomato can't be bigger than 9 cm any axis"""
    """ default filters assume tomato can't be smaller than 4 cm any axis"""
    picked_df = tracks_df.query('is_picked == 1')
    tracks_width = picked_df.groupby('track_id').width.mean()
    tracks_height = picked_df.groupby('track_id').height.mean()

    tracks_width = tracks_width[tracks_width < upper_filter].copy()
    tracks_width = tracks_width[tracks_width > lower_filter].copy()

    tracks_height = tracks_height[tracks_height < upper_filter].copy()
    tracks_height = tracks_height[tracks_height > lower_filter].copy()

    return tracks_width, tracks_height

def get_color_bins(tracks_to_color):

    color_bins = {}
    colors = []
    for color in tracks_to_color.values():
        colors.append(color)

    unique_colors, color_counts = np.unique(colors, return_counts=True)

    for bin in range(1, 6): # we have colors in range 1 to 5 include
        if bin in unique_colors:
            bin_index = np.argwhere(unique_colors == bin)
            color_bins[bin] = color_counts[bin_index[0, 0]]
        else:
            color_bins[bin] = 0

    return color_bins

def convert_dist(data, mu_orig, std_orig, mu_target, std_target):

    z_score = (data - mu_orig) / std_orig
    converted_data = z_score * std_target + mu_target

    return converted_data

def apply_model(width, height, coef, intercept):

    return width * coef[0] + height * coef[1] + intercept

def get_weight_std(conv_width, conv_height, model_coef):
    var_width = np.var(conv_width)
    var_height = np.var(conv_height)
    weight_std = np.sqrt((var_width * np.power(model_coef[0], 2)) + (var_height * np.power(model_coef[1], 2)))

    return weight_std

def create_results_dict(gt_data, tracks_to_color, color_bins, weight, weight_std, conv_width, row, rep):

    plot_results = dict()
    plot_results['Farm'] = gt_data['Farm'].values[0]
    plot_results['Plot_name'] = gt_data['Plot_name'].values[0]
    plot_results['FS_variety'] = gt_data['FS_variety'].values[0]
    plot_results['count'] = len(tracks_to_color)
    plot_results['bin_1'] = color_bins[1]
    plot_results['bin_2'] = color_bins[2]
    plot_results['bin_3'] = color_bins[3]
    plot_results['bin_4'] = color_bins[4]
    plot_results['bin_5'] = color_bins[5]
    plot_results['total_weight_kg'] = np.round((len(tracks_to_color) * np.mean(weight)) / 1000, 1)  # in kg
    plot_results['weight_avg_gr'] = np.round(weight, 1)
    plot_results['weight_std'] = np.round(weight_std, 1)
    plot_results['size_avg_mm'] = np.round(np.mean(conv_width), 1)
    plot_results['size_std'] = np.round(np.std(conv_width), 1)
    plot_results['row'] = row
    plot_results['rep'] = rep

    return plot_results






if __name__ == '__main__':

    folder_path = "/home/fruitspec-lab/FruitSpec/Data/Syngenta/110124"
    gt_data_path = "/home/fruitspec-lab/FruitSpec/Data/Syngenta/almeria_plot_meta.csv"
    if gt_data_path is not None:
        gt_df = pd.read_csv(gt_data_path)
    else:
        gt_df = None

    to_debug = True
    min_samp = 3
    dist_center_threshold = 200
    color_break_threshold = 0.9

    model_coef = [4.25467987, 2.40293413]
    model_intercept = -271.53407231486557

    gt_width_mean = 66.64207831325301
    gt_width_std = 6.524271084973641
    gt_height_mean = 54.35472891566266
    gt_height_std = 6.012594102523396

    counter_width_mean = 67.55363882906708
    counter_width_std = 12.210693006488093
    counter_height_mean = 67.61603085290156
    counter_height_std = 10.08302167961458

    rows = os.listdir(folder_path)
    #rows = ['row_2', 'row_6', 'row_7', 'row_8']
    res = []
    width = []
    height = []
    for row in tqdm(rows):
        row_path = os.path.join(folder_path, row)
        if not os.path.isdir(row_path):
            continue
        repetitions = os.listdir(row_path)
        row_number = int(row.split('_')[-1])
        direction = 'right' if row_number % 2 != 0 else 'left'
        for rep in repetitions:
            rep_path = os.path.join(row_path, rep)
            tracks_path = os.path.join(rep_path, 'tracks.csv')
            slice_json_path = os.path.join(rep_path, 'zed_slice_data.json')

            if not os.path.exists(slice_json_path):
                continue
            try:

                tracks_df, slices_df = read_tracks_and_slices(tracks_path, slice_json_path, direction)

                row_results, trees_tracks = count_trees_fruits(tracks_df, slices_df, frame_width=1080)
                trees = list(trees_tracks.keys())
                for t in trees:
                    section_df = trees_tracks[t]
                    if to_debug:
                        cluster_path = os.path.join(rep_path, 'tree_cluster_c3_2', str(t))
                        tree_path = os.path.join(rep_path, 'tree_c3_2', str(t))
                        validate_output_path(cluster_path)
                        validate_output_path(tree_path)


                        debug = {'path': rep_path,
                                 'tree_id': t,
                                 'picked_col': -1,
                                 'cluster_col': -5,
                                 'cluster_output': cluster_path,
                                 'tree_output': tree_path
                                 }
                    else:
                        debug = None



                    picked_clusters, color_bins, tracks_to_color,  tracks_width, tracks_height = analyze_section(section_df,
                                                                                                                 min_samp=min_samp,
                                                                                                                 dist_center_threshold=dist_center_threshold,
                                                                                                                 color_break_threshold=color_break_threshold,
                                                                                                                 debug=debug)

                    width = np.array(tracks_width) * 1000  # convert to mm
                    height = np.array(tracks_height) * 1000  # convert to mm

                    conv_width = convert_dist(width,
                                              counter_width_mean,
                                              counter_width_std,
                                              gt_width_mean,
                                              gt_width_std)
                    #print(f'{row}, {rep} width samples: {len(conv_width)}')

                    conv_height = convert_dist(height,
                                              counter_height_mean,
                                              counter_height_std,
                                              gt_height_mean,
                                              gt_height_std)

                    #print(f'{row}, {rep} height samples: {len(conv_height)}')

                    mean_width = np.nanmean(conv_width)
                    mean_height = np.nanmean(conv_height)

                    weight = apply_model(mean_width, mean_height, model_coef, model_intercept)
                    weight_std = get_weight_std(conv_width, conv_height, model_coef)

                    gt_data = gt_df.query(f'row == {row_number} and tree == {t}')
                    #gt_data = gt_df.query(f'Plot_name == "{plot_name}"')

                    section_results = create_results_dict(gt_data,
                                                          tracks_to_color,
                                                          color_bins,
                                                          weight,
                                                          weight_std,
                                                          conv_width,
                                                          row,
                                                          rep)



                    res.append(section_results)
            except:
                print(f'failed to run {rep_path}')


    results = pd.DataFrame(res, columns=list(section_results.keys()))
    results.to_csv(os.path.join(folder_path, 'results_a.csv'))

