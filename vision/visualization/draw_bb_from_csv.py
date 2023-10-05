import os
import cv2
import pandas as pd
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf

from vision.pipelines.ops.frame_loader import FramesLoader
from vision.misc.help_func import validate_output_path, get_repo_dir
from vision.data.results_collector import ResultsCollector



def validate_from_files(tracks, cfg, args, alignment=None, jai_only=False, data_index=7):
    print('arranging data')
    dets = track_to_det(tracks)
    jai_frames = list(dets.keys())
    if alignment is not None:
        a_hash = get_alignment_hash(alignment)
    elif not jai_only:
        print('Not enough alignment data to draw alignment results')
        return

    frame_loader = FramesLoader(cfg, args)
    frame_loader.batch_size = 1

    for id_ in tqdm(jai_frames):
        zed_batch, depth_batch, jai_batch, rgb_batch = frame_loader.get_frames(int(id_), 0)
        if jai_only:
            rc = ResultsCollector()
            jai = rc.draw_dets(frame = jai_batch[0], dets = dets[id_], t_index=data_index)
            fp = os.path.join(args.output_folder, 'Dets')
            validate_output_path(fp)

            cv2.imwrite(os.path.join(fp, f"dets_f{id_}.jpg"), jai)
        else:
            save_aligned(zed_batch[0],
                         jai_batch[0],
                         args.output_folder,
                         id_,
                         corr=a_hash[id_], dets=dets[id_])


def save_aligned(zed, jai, output_folder, f_id, corr=None, sub_folder='FOV',dets=None):
    if corr is not None and np.sum(np.isnan(corr)) == 0:
        zed = zed[int(corr[1]):int(corr[3]), int(corr[0]):int(corr[2]), :]

    gx = 680 / jai.shape[1]
    gy = 960 / jai.shape[0]
    zed = cv2.resize(zed, (680, 960))
    jai = cv2.resize(jai, (680, 960))

    if dets is not None:
        dets = np.array(dets)
        dets[:, 0] = dets[:, 0] * gx
        dets[:, 2] = dets[:, 2] * gx
        dets[:, 1] = dets[:, 1] * gy
        dets[:, 3] = dets[:, 3] * gy
        jai = ResultsCollector.draw_dets(jai, dets, t_index=7, text=False)
        zed = ResultsCollector.draw_dets(zed, dets, t_index=7, text=False)

    canvas = np.zeros((960, 680*2, 3))
    canvas[:, :680, :] = zed
    canvas[:, 680:, :] = jai

    fp = os.path.join(output_folder, sub_folder)
    validate_output_path(fp)
    cv2.imwrite(os.path.join(fp, f"aligned_f{f_id}.jpg"), canvas)



def track_to_det(tracks_df):
    dets = {}
    for i, row in tracks_df.iterrows():
        if row['frame_id'] in list(dets.keys()):
            dets[int(row['frame_id'])].append([row['x1'], row['y1'], row['x2'], row['y2'], row['obj_conf'], row['class_conf'], int(row['frame_id']), 0])
        else:
            dets[int(row['frame_id'])] = [[row['x1'], row['y1'], row['x2'], row['y1'], row['obj_conf'], row['class_conf'], int(row['frame_id']), 0]]


    return dets

def get_alignment_hash(alignment):
    data = alignment.to_numpy()
    frames = data[:, 6]
    corr = data[:, :4]

    hash = {}
    for i in range(len(frames)):
        hash[frames[i]] = list(corr[i, :])

    return hash


if __name__ == "__main__":
    repo_dir = get_repo_dir()
    pipeline_config = "/vision/pipelines/config/pipeline_config.yaml"
    runtime_config = "/vision/pipelines/config/dual_runtime_config.yaml"
    cfg = OmegaConf.load(repo_dir + pipeline_config)
    args = OmegaConf.load(repo_dir + runtime_config)

    folder = "/home/fruitspec-lab-3/FruitSpec/Data/customers/MOTCHA/RAISTENB/060723/row_2/1"
    args.sync_data_log_path = os.path.join(folder, "jaized_timestamps.csv")
    args.jai.movie_path = os.path.join(folder, 'Result_FSI.mkv')
    args.zed.movie_path = os.path.join(folder, 'ZED.mkv')
    args.depth.movie_path = os.path.join(folder, 'ZED_Y.mkv')
    args.rgb_jai.movie_path = os.path.join(folder, 'Result_RGB.mkv')
    tracks_p = os.path.join(folder, "tracks.csv")
    alignment_p = os.path.join(folder, "alignment.csv")

    tracks = pd.read_csv(tracks_p)
    alignment = pd.read_csv(alignment_p)
    data_index = 7 # which column to use to detrmine bbox color
    args.output_folder = "/home/fruitspec-lab-3/FruitSpec/Data/customers/MOTCHA/RAISTENB/060723/row_2/1/adt_yoloX"
    validate_output_path(args.output_folder)
    validate_from_files(tracks=tracks, cfg=cfg, args=args, alignment=alignment, jai_only=True, data_index=data_index)

    print('Done')