from vision.pipelines.rgb_pipeline import get_repo_dir, run as run_rgb_pipeline, validate_output_path
import os
from omegaconf import OmegaConf
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import time

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
