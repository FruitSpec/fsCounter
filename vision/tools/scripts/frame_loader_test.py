from omegaconf import OmegaConf
from vision.pipelines.adt_pipeline import Pipeline
from vision.misc.help_func import get_repo_dir, validate_output_path
import numpy as np
from tqdm import tqdm
import time

def load_frame(cam, n_frames=25):

    for i in range(n_frames):
        s = time.time()
        _, f = cam.get_frame()
        _, f = cam.get_frame()
        _, f = cam.get_frame()
        _, f = cam.get_frame()
        e = time.time()
        print(f"loading frame time: {e-s:.4f}")

if __name__ == "__main__":
    repo_dir = get_repo_dir()
    pipeline_config = "/vision/pipelines/config/pipeline_config.yaml"
    runtime_config = "/vision/pipelines/config/dual_runtime_config.yaml"
    cfg = OmegaConf.load(repo_dir + pipeline_config)
    args = OmegaConf.load(repo_dir + runtime_config)
    args.zed.movie_path = args.rgb_jai.movie_path # to use mkv
    args.depth.movie_path = "/home/mic-730ai/fruitspec/test_data/grayscale/DEPTH-QUALITY/DEPTH_1.mkv"
    args.depth.frame_size = args.zed.frame_size
    args.zed.frame_size = args.rgb_jai.frame_size
    args.sync_data_log_path = "/home/mic-730ai/fruitspec/test_data/validate_refactor/sync_data/frame_loader_sync_mode/jaized_timestamps_2.log"
    cfg.frame_loader.mode = 'sync_mkv'


    validate_output_path(args.output_folder)
    p = Pipeline(cfg, args)
    p.frames_loader.sync_zed_ids = np.arange(0, 100)
    p.frames_loader.sync_jai_ids = np.arange(10, 110)
    for i in tqdm(range(100)):
        s = time.time()
        zed_batch, depth_batch, jai_batch, rgb_jai_batch = p.get_frames(i)
        e = time.time()
        print(f"iteration time {e-s:.4f}")



