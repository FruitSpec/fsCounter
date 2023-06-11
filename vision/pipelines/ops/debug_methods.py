import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


from vision.misc.help_func import validate_output_path


def plot_alignmnt_graph(args, results_collector, frame_drop_jai, show=True):
    run_type = args.debug.run_type
    tx_data = [a["tx"] for a in results_collector.alignment]
    frames = [a["frame"] for a in results_collector.alignment]
    zed_shifts = np.array([a["zed_shift"] for a in results_collector.alignment])
    plt.figure(figsize=(15, 10))
    tx_data = np.clip(tx_data, -50, 200)
    plt.plot(frames, tx_data)
    graph_std = np.round(np.std(tx_data), 2)
    tx_conv = np.convolve(tx_data, np.ones(10) / 10, mode='same')
    conv_noise = np.round(np.mean(np.abs(tx_data[5:-5] - tx_conv[5:-5])), 2)
    conv_noise_med = np.round(np.median(np.abs(tx_data[5:-5] - tx_conv[5:-5])), 2)
    plt.plot(frames, tx_conv, color="orange")
    block = os.path.basename(os.path.dirname(args.output_folder))
    row = os.path.basename(args.output_folder)
    plt.title(f"{block}-{row}-{run_type}_std:{graph_std}_conv_noise:({conv_noise},{conv_noise_med})")
    for frame in frame_drop_jai[frame_drop_jai < np.max(frames)]:
        plt.vlines(frame, np.min(tx_data), np.max(tx_data), color="red", linestyles="dotted")
    for frame in np.array(frames)[np.where(zed_shifts[1:] - zed_shifts[:-1] == 1)[0]]:
        plt.vlines(frame, np.min(tx_data), np.max(tx_data), color="green", linestyles="dotted")
    plt.savefig(f"{args.output_folder}_{run_type}_graph.png")
    if show:
        plt.show()


def draw_on_tracked_imgaes(args, slice_df, filtered_trees, jai_cam, results_collector):
    """
    this function draws tracking and slicing on an imgae, and saves each tree to args.debug_folder
    :param args: arguments conifg file
    :param slice_df: a slices data frame containing [frame_id, tree_id, start, end]
    :param filtered_trees: a dataframe of tracking results
    :param jai_cam: jai vamera video wrapper object
    :param results_collector: result collector object
    :return: None
    """
    slice_df["start"] = slice_df["start"].replace(-1, 0)
    slice_df["end"] = slice_df["end"].replace(-1, int(jai_cam.get_height() - 1))
    for tree_id in filtered_trees["tree_id"].unique():
        tree_slices = slice_df[slice_df["tree_id"] == tree_id]
        tree_tracks = filtered_trees[filtered_trees["tree_id"] == tree_id]
        unique_tracks = tree_tracks["track_id"].unique()
        new_ids = dict(zip(unique_tracks, range(len(unique_tracks))))
        tree_tracks.loc[:, "track_id"] = tree_tracks["track_id"].map(new_ids)
        for f_id in tree_tracks["frame_id"].unique():
            dets = tree_tracks[tree_tracks["frame_id"] == f_id]
            dets  = dets.values
            frame_slices = tree_slices[tree_slices["frame_id"] == f_id][["start", "end"]].astype(int).values[0]
            debug_outpath = os.path.join(args.debug_folder, f"{args.block_name}_{args.row_name}_T{tree_id}")
            validate_output_path(debug_outpath)
            ret, frame = jai_cam.get_frame(f_id)
            if ret:
                frame = cv2.line(frame, (frame_slices[0], 0), (frame_slices[0], int(jai_cam.get_width())),
                                 color=(255, 0, 0), thickness=2)
                frame = cv2.line(frame, (frame_slices[1], 0), (frame_slices[1], int(jai_cam.get_width())),
                                 color=(255, 0, 0), thickness=2)
                results_collector.draw_and_save(frame, dets, f_id, debug_outpath, t_index=6)
