import os
import sys
from omegaconf import OmegaConf
from tqdm import tqdm


from vision.misc.help_func import get_repo_dir, load_json, validate_output_path

repo_dir = get_repo_dir()
sys.path.append(os.path.join(repo_dir, 'vision', 'detector', 'yolo_x'))

from vision.pipelines.ops.frame_loader import FramesLoader
from vision.data.results_collector import ResultsCollector
from vision.tools.translation import translation as T



def run(cfg, args, n_frames=200):
    cfg.batch_size = 1
    print(f'Inferencing on {args.jai.movie_path}\n')
    rc = ResultsCollector(rotate=args.rotate)

    frame_loader = FramesLoader(cfg, args)
    translation = T(cfg.translation.translation_size, cfg.translation.dets_only, cfg.translation.mode)

    f_id = 0
    n_frames = len(frame_loader.sync_zed_ids) if n_frames is None else min(n_frames, len(frame_loader.sync_zed_ids))
    pbar = tqdm(total=n_frames)
    data = []
    while f_id < n_frames:
        pbar.update(cfg.batch_size)
        zed_batch, depth_batch, jai_batch, rgb_batch = frame_loader.get_frames(f_id, 0)
        if f_id < 105:
            f_id+= 1
            continue
        debug = []
        output = os.path.join(args.output_folder, 'translation')
        validate_output_path(output)
        for i in range(cfg.batch_size):
            debug.append({'output_path': output, 'f_id': f_id + i})

        translation.batch_translation(jai_batch, [[],[],[],[]], debug)

        f_id += 1


if __name__ == "__main__":
    repo_dir = get_repo_dir()
    pipeline_config = "/vision/pipelines/config/pipeline_config.yaml"
    runtime_config = "/vision/pipelines/config/dual_runtime_config.yaml"
    cfg = OmegaConf.load(repo_dir + pipeline_config)
    args = OmegaConf.load(repo_dir + runtime_config)

    #folder = "/media/matans/My Book/FruitSpec/NWFMXX/G10000XX/070623/row_12/1"
    folder = "/media/matans/My Book/FruitSpec/Customers_data/Fowler/daily/BLOCK700/200723/row_4/1"
    args.zed.movie_path = os.path.join(folder, "ZED.mkv")
    args.depth.movie_path = os.path.join(folder, "DEPTH.mkv")
    args.jai.movie_path = os.path.join(folder, "Result_FSI.mkv")
    args.rgb_jai.movie_path = os.path.join(folder, "Result_RGB.mkv")
    args.sync_data_log_path = os.path.join(folder, "jaized_timestamps.csv")
    args.output_folder = os.path.join("/home/matans/Documents/fruitspec/sandbox/sa_vs_lg", 'BLOCK700_row4_fix')
    validate_output_path(args.output_folder)

    run(cfg, args, 200)
