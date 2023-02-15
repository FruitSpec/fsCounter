from vision.pipelines.rgb_pipeline import get_repo_dir, run as run_rgb_pipeline, validate_output_path
import os
from omegaconf import OmegaConf

def run_real_time():
    repo_dir = get_repo_dir()
    pipeline_config = repo_dir + "vision/pipelines/config/pipeline_config.yaml"
    runtime_config = repo_dir + "/vision/pipelines/config/runtime_config.yaml"
    cfg = OmegaConf.load(pipeline_config)
    args = OmegaConf.load(runtime_config)
    movies_path = args.movie_path
    analysis_path = args.output_folder
    for scan in os.listdir(movies_path):
        for row in os.listdir(os.path.join(movies_path, scan)):
            args.movie_path = os.path.join(movies_path, scan, row, [i for i in os.listdir(os.path.join(movies_path, scan, row)) if i.endswith('.svo')][0])
            validate_output_path(os.path.join(analysis_path, scan))
            validate_output_path(os.path.join(analysis_path, scan, row))
            args.output_folder = os.path.join(analysis_path, scan, row)
            run_rgb_pipeline(cfg, args)