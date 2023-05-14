from omegaconf import OmegaConf
from vision.pipelines.adt_pipeline import Pipeline
from vision.misc.help_func import get_repo_dir, validate_output_path

if __name__ == "__main__":
    repo_dir = get_repo_dir()
    pipeline_config = "/vision/pipelines/config/pipeline_config.yaml"
    runtime_config = "/vision/pipelines/config/dual_runtime_config.yaml"
    cfg = OmegaConf.load(repo_dir + pipeline_config)
    args = OmegaConf.load(repo_dir + runtime_config)
    args.zed.movie_path = args.rgb_jai.movie_path # to use mkv
    args.zed.frame_size = args.rgb_jai.frame_size
    args.depth.movie_path = args.jai.movie_path  # to use mkv
    args.depth.frame_size = args.jai.frame_size
    cfg.frame_loader.mode = 'sync_mkv'

    validate_output_path(args.output_folder)
    p = Pipeline(cfg, args)

