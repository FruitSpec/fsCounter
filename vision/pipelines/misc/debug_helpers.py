import os
import cv2
import numpy as np

from vision.data.results_collector import ResultsCollector
from vision.misc.help_func import validate_output_path

def save_aligned(zed, jai, output_folder, f_id, corr=None, sub_folder='FOV',dets=None, lense=61, name=None, index_=6, thick=2):
    if lense != 83:
        if corr is not None and np.sum(np.isnan(corr)) == 0:
            zed = zed[int(corr[1]):int(corr[3]), int(corr[0]):int(corr[2]), :]

    gx = 680 / jai.shape[1]
    gy = 960 / jai.shape[0]
    zed = cv2.resize(zed, (680, 960))
    jai = cv2.resize(jai, (680, 960))

    if dets is not None and len(dets) > 0:
        dets = np.array(dets)
        dets[:, 0] = dets[:, 0] * gx
        dets[:, 2] = dets[:, 2] * gx
        dets[:, 1] = dets[:, 1] * gy
        dets[:, 3] = dets[:, 3] * gy
        jai = ResultsCollector.draw_dets(jai, dets, t_index=index_, text=False, thickness=thick)
        zed = ResultsCollector.draw_dets(zed, dets, t_index=index_, text=False, thickness=thick)

    canvas = np.zeros((960, 680*2, 3))
    canvas[:, :680, :] = zed
    canvas[:, 680:, :] = jai

    fp = os.path.join(output_folder, sub_folder)
    validate_output_path(fp)
    if name is None:
        name = os.path.join(fp, f"aligned_f{f_id}.jpg")
    else:
        name = os.path.join(fp, f"{name}_f{f_id}.jpg")
    cv2.imwrite(name, canvas)
