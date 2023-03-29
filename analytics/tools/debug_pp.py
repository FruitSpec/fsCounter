import os
from tqdm import tqdm

from vision.tools.video_wrapper import video_wrapper
from vision.data.results_collector import ResultsCollector
from vision.misc.help_func import validate_output_path


def debug_plots(df, raw_path, output_path):
    for scan, scan_df in df.groupby('scan'):
        for row, row_df in scan_df.groupby('row'):
            row_path = os.path.join(raw_path, scan, row)
            if not os.path.exists(row_path):
                continue
            movie_path = os.path.join(row_path, [i for i in os.listdir(os.path.join(row_path)) if
                                                 i.endswith('.svo')][0])

            output_folder = os.path.join(output_path, scan, row)
            cam = video_wrapper(movie_path, 2, 0.1, 2.5)  # hard coded
            results_collector = ResultsCollector(rotate=2)  # hard coded
            # Read until video is completed
            number_of_frames = cam.get_number_of_frames()
            color_ls = [(0, 64, 255), (194, 24, 7)]

            f_id = 0
            pbar = tqdm(total=number_of_frames)
            while True:
                pbar.update(1)
                frame, depth, point_cloud = cam.get_zed()
                if not cam.res:  # couldn't get frames
                    #     Break the loop
                    break
                frame_results = row_df[row_df['frame'] == f_id]

                flag = 0
                prev_plot_id = None
                for plot_id, df_plot in frame_results.groupby('plot_id'):
                    # switch every plot
                    if prev_plot_id == None or prev_plot_id == plot_id:
                        flag = ~flag
                    color = color_ls[flag]
                    dets = df_plot[['x1', 'y1', 'x2', 'y2', 'track_id']].values.tolist()
                    validate_output_path(os.path.join(output_folder, 'trk_results'))
                    results_collector.draw_and_save(frame.copy(), dets, f_id, os.path.join(output_folder, 'trk_results'), t_index=4, color=color)
                    prev_plot_id = plot_id
                f_id += 1

            # When everything done, release the video capture object
            cam.close()
