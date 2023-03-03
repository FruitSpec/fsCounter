from vision.pipelines.rgb_pipeline import get_repo_dir, run as run_rgb_pipeline, validate_output_path
import os
from omegaconf import OmegaConf
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time


def multi_process_wrapper(cfg, args, movies_path, scan, row, analysis_path):
    try:
        args.movie_path = os.path.join(movies_path, scan, row,
                                       [i for i in os.listdir(os.path.join(movies_path, scan, row)) if
                                        i.endswith('.svo')][0])
    except:
        print(f'{os.path.join(movies_path, scan, row)} SVO not found')
        return

    # try:
    validate_output_path(os.path.join(analysis_path, scan))
    validate_output_path(os.path.join(analysis_path, scan, row))
    # set output path
    args.output_folder = os.path.join(analysis_path, scan, row)
    run_rgb_pipeline(cfg, args)



def run_real_time(max_workers=3, pipeline_config=""):
    """
    rgb pipeline on multiple files
    parallel running according max_worker and real time pipeline config
    """
    # vision.pipeline configuration
    repo_dir = get_repo_dir()
    if pipeline_config == "":
        pipeline_config = repo_dir + "/vision/pipelines/config/pipeline_config.yaml"
    runtime_config = repo_dir + "/vision/pipelines/config/runtime_config.yaml"
    cfg = OmegaConf.load(pipeline_config)
    args = OmegaConf.load(runtime_config)

    # analytics configuration
    offline_config = OmegaConf.load(os.getcwd() + '/config/runtime.yml')
    movies_path = offline_config.video_path
    analysis_path = offline_config.output_path

    scans, rows = [], []
    for scan in os.listdir(movies_path):
        for row in os.listdir(os.path.join(movies_path, scan)):
            scans.append(scan)
            rows.append(row)
    n_files = len(scans)
    cfgs = [cfg] * n_files
    args_list = [args] * n_files
    movies_paths = [movies_path] * n_files
    analysis_paths = [analysis_path] * n_files
    s_t = time.time()
    # with ThreadPoolExecutor(max_workers=max_workers) as executor:
    #     results = list(executor.map(multi_process_wrapper, cfgs, args_list, movies_paths, scans, rows, analysis_paths))
    if max_workers == 1:
        for i in range(n_files):
            multi_process_wrapper(cfgs[i], args_list[i], movies_paths[i], scans[i], rows[i], analysis_paths[i])
    else:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            executor.map(multi_process_wrapper, cfgs, args_list, movies_paths, scans, rows, analysis_paths)
    print("total time: ", time.time()-s_t)


def run_real_time():
    """
    rgb pipeline on multiple files
    serial running , single file
    """
    # vision.pipeline configuration
    repo_dir = get_repo_dir()
    pipeline_config = repo_dir + "/vision/pipelines/config/pipeline_config.yaml"
    runtime_config = repo_dir + "/vision/pipelines/config/runtime_config.yaml"
    cfg = OmegaConf.load(pipeline_config)
    args = OmegaConf.load(runtime_config)

    # analytics configuration
    offline_config = OmegaConf.load(os.getcwd() + '/config/runtime.yml')
    movies_path = offline_config.video_path
    analysis_path = offline_config.output_path

    for scan in os.listdir(movies_path):
        for row in os.listdir(os.path.join(movies_path, scan)):
            try:
                args.movie_path = os.path.join(movies_path, scan, row,
                                               [i for i in os.listdir(os.path.join(movies_path, scan, row)) if
                                                i.endswith('.svo')][0])
            except:
                print(f'{os.path.join(movies_path, scan, row)} SVO not found')

                # validation
            validate_output_path(os.path.join(analysis_path, scan))
            validate_output_path(os.path.join(analysis_path, scan, row))
            # set output path
            args.output_folder = os.path.join(analysis_path, scan, row)

            run_rgb_pipeline(cfg, args)
