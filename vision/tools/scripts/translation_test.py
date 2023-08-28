import os
import sys

import cv2
import pandas as pd
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
    translation = T(cfg.batch_size, cfg.translation.translation_size, cfg.translation.dets_only, cfg.translation.mode)

    f_id = 0
    n_frames = len(frame_loader.sync_zed_ids) if n_frames is None else min(n_frames, len(frame_loader.sync_zed_ids))
    pbar = tqdm(total=n_frames)
    data = []
    while f_id < n_frames:
        pbar.update(cfg.batch_size)
        zed_batch, depth_batch, jai_batch, rgb_batch = frame_loader.get_frames(f_id, 0)
        #if f_id < 105:
        #    f_id += 1
        #    continue
        debug = []
        output_translation = os.path.join(args.output_folder, 'translation')
        validate_output_path(output_translation)
        output_frames = os.path.join(args.output_folder, 'frames')
        validate_output_path(output_frames)
        for i in range(cfg.batch_size):
            debug.append({'output_path': output_translation, 'f_id': f_id + i})
            cv2.imwrite(os.path.join(output_frames, f'frame_f{f_id + i}.jpg'), jai_batch[i])

        res = translation.batch_translation(jai_batch, [[],[],[],[]], debug)

        data.append({'tx': res[0][0], 'ty': res[0][1], 'score': res[0][2]})
        f_id += 1

    df = pd.DataFrame(data, columns=['tx', 'ty', 'score'])
    df.to_csv(os.path.join(output, 'translation.csv'))

if __name__ == "__main__":
    repo_dir = get_repo_dir()
    pipeline_config = "/vision/pipelines/config/pipeline_config.yaml"
    runtime_config = "/vision/pipelines/config/dual_runtime_config.yaml"
    cfg = OmegaConf.load(repo_dir + pipeline_config)
    args = OmegaConf.load(repo_dir + runtime_config)

    #folder = "/media/matans/My Book/FruitSpec/NWFMXX/G10000XX/070623/row_12/1"
    #folder = "/media/matans/My Book/FruitSpec/Customers_data/Fowler/daily/FREDIANI/210723/row_5/1"
    folder = "/media/matans/My Book/FruitSpec/Customers_data/Fowler/daily/BLOCK700/200723/row_4/1"
    output = "/home/matans/Documents/fruitspec/sandbox/translation"
    validate_output_path(output)
    args.zed.movie_path = os.path.join(folder, "ZED.mkv")
    args.depth.movie_path = os.path.join(folder, "DEPTH.mkv")
    args.jai.movie_path = os.path.join(folder, "Result_FSI.mkv")
    args.rgb_jai.movie_path = os.path.join(folder, "Result_RGB.mkv")
    args.sync_data_log_path = os.path.join(folder, "jaized_timestamps.csv")
    #args.output_folder = os.path.join(output, 'FREDIANI_210723_row_5_1_all_image')
    args.output_folder = os.path.join(output, 'BLOCK700_row_4_sqdiff')
    validate_output_path(args.output_folder)

    run(cfg, args, 200)

