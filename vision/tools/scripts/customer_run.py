import os
from omegaconf import OmegaConf

from vision.misc.help_func import get_repo_dir, validate_output_path
from vision.pipelines.adt_pipeline import run
from vision.pipelines.ops.distance_slicer import slice_row



if __name__ == "__main__":
    repo_dir = get_repo_dir()
    pipeline_config = "/vision/pipelines/config/pipeline_config.yaml"
    runtime_config = "/vision/pipelines/config/dual_runtime_config.yaml"
    #fe_config = "/vision/feature_extractor/feature_extractor_config.yaml"

    cfg = OmegaConf.load(repo_dir + pipeline_config)
    args = OmegaConf.load(repo_dir + runtime_config)
#    fe_args = OmegaConf.load(repo_dir + fe_config)

    zed_name = "ZED.mkv"
    depth_name = "DEPTH.mkv"
    fsi_name = "Result_FSI.mkv"
    rgb_name = "Result_RGB.mkv"
    time_stamp = "jaized_timestamps.csv"

    nav_folder = "/media/matans/My Book/FruitSpec/sandbox/tracker/depth_adaptive"
    nav_list = os.listdir(nav_folder)
    output_path = "/media/matans/My Book/FruitSpec/sandbox/tracker/depth_adaptive"
    validate_output_path(output_path)
    plots_dir = "/media/matans/My Book/FruitSpec/Mehadrin"
    plots = os.listdir(plots_dir)
    #rows = ["/home/matans/Documents/fruitspec/sandbox/NWFM/val"]
    for plot in plots:
        plot_folder = os.path.join(plots_dir, plot)
        if os.path.isdir(plot_folder):
            cur_output = os.path.join(output_path, plot)
            validate_output_path(cur_output)
            dates = os.listdir(plot_folder)
            for date in dates:
                date_folder = os.path.join(plot_folder, date)
                if os.path.isdir(date_folder):
                    cur_output = os.path.join(output_path, plot, date)
                    validate_output_path(cur_output)
                    rows = os.listdir(date_folder)
                    for row in rows:
                        row_folder = os.path.join(date_folder, row, '1')
                        if os.path.isdir(row_folder):
                            cur_output = os.path.join(output_path, plot, date, row)
                            validate_output_path(cur_output)
                            cur_output = os.path.join(output_path, plot, date, row, '1')
                            validate_output_path(cur_output)



                            args.output_folder = cur_output
                            args.sync_data_log_path = os.path.join(row_folder, time_stamp)
                            if not os.path.exists(args.sync_data_log_path):
                                continue
                            args.zed.movie_path = os.path.join(row_folder, zed_name)
                            if not os.path.exists(args.zed.movie_path):
                                continue
                            args.depth.movie_path = os.path.join(row_folder, depth_name)
                            if not os.path.exists(args.depth.movie_path):
                                continue
                            args.jai.movie_path = os.path.join(row_folder, fsi_name)
                            if not os.path.exists(args.jai.movie_path):
                                continue
                            args.rgb_jai.movie_path = os.path.join(row_folder, rgb_name)
                            if not os.path.exists(args.rgb_jai.movie_path):
                                continue

                            validate_output_path(args.output_folder)

                            nav_path = None
                            for nav in nav_list:
                                if date in nav:
                                    nav_path = os.path.join(nav_folder, nav)
                                    break

                            try:
                                 if not os.path.exists(os.path.join(cur_output, 'tracks.csv')):

                                     rc = run(cfg, args, n_frames=150)
                                     rc.dump_feature_extractor(args.output_folder)
                                 else:
                                     # Done running on the row in previous run, skip
                                     print(f'tracks exist. skipping row {row}')

                                 # if (nav_path is not None) and \
                                 #        (not os.path.exists(os.path.join(cur_output, 'slices.csv'))) \
                                 #        and (os.path.exists(os.path.join(row_folder, "jai_translation.csv"))):
                                 #
                                 # #if (nav_path is not None) and \
                                 # #        (os.path.exists(os.path.join(row_folder, "jai_translation.csv"))):
                                 #     slices_df = slice_row(row_folder, nav_path)
                                 #     slices_df.to_csv(os.path.join(cur_output, 'slices.csv'))
                                 #     print('Done Slicing')
                                 # else:
                                 #     print('Slices exist')

                                #if not os.path.exists(os.path.join(cur_output, 'features.csv')):
                                #    fe_args, adt_args, slices = update_run_args(fe_args, args, row_folder)
                                #    features_df = run_fe(cfg, fe_args, adt_args, slices, row)
                                    #
                                #    features_df.to_csv(os.path.join(cur_output, 'features.csv'))
                                #    print('Done feature extraction')
                            except:
                                print(f'failed {row_folder}')


