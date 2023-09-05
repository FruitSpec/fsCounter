import os
from tqdm import tqdm
import numpy as np

import pandas as pd

from vision.tracker.fsTracker.score_func import calc_iou
from vision.data.COCO_utils import load_coco_file

def analyze(src_path, res_path):

    ious = np.arange(0.3, 0.95, 0.05)
    confs = np.arange(0.05, 0.95, 0.01)

    src = load_coco_file(src_path)
    res = load_coco_file(res_path)

    src_to_res_ids = get_src_to_res_ids(src, res)
    res_mat = np.zeros((len(confs), len(ious), len(src_to_res_ids), 3))  # C: 0 - precision, 1 - recall, 2 - f1

    src_id_to_bbox = get_id_to_anns(src)
    res_id_to_bbox = get_id_to_anns(res)
    src_id_list = list(src_id_to_bbox.keys())
    res_id_list = list(res_id_to_bbox.keys())

    for img_id, (src_id, res_id) in tqdm(enumerate(src_to_res_ids.items())):
        if src_id in src_id_list:
            src_anns = src_id_to_bbox[src_id]
        else:
            continue
        if res_id in res_id_list:
            res_anns = res_id_to_bbox[res_id]
        else:
            continue


        src_bbox, src_conf = anns_to_vecs(src_anns)
        res_bbox, res_conf = anns_to_vecs(res_anns)


        iou = calc_iou(src_bbox, res_bbox)
        conf_mat = np.expand_dims(src_conf, axis=1) * np.transpose(np.expand_dims(res_conf, axis=1))

        for conf_id, conf in enumerate(confs):
            for iou_id, test_iou in enumerate(ious):
                bool_conf = conf_mat > conf
                bool_iou = iou > test_iou
                bool_ = bool_conf & bool_iou
                total_predicted = np.sum(res_conf > conf)

                TP = np.sum(bool_)
                FP = total_predicted - TP
                FN = len(src_bbox) - TP

                precision = TP / max(total_predicted, 1)
                recall = TP / len(src_bbox)
                f1 = (2 * TP) / ((2 * TP) + FN + FP)

                res_mat[conf_id, iou_id, img_id, 0] = precision
                res_mat[conf_id, iou_id, img_id, 1] = recall
                res_mat[conf_id, iou_id, img_id, 2] = f1


    return res_mat, ious, confs







def anns_to_vecs(anns):

    bboxes = []
    confs = []
    for ann in anns:
        ann_keys = list(ann.keys())
        bboxes.append(ann['bbox'])

        if 'score' in ann_keys:
            confs.append(ann['score'])
        else:
            confs.append(1)

    return bboxes, confs


def get_src_to_res_ids(src, res):

    src_imgs = src['images'].copy()
    res_imgs = res['images'].copy()

    hash_ = dict()
    for s_img in src_imgs:
        for r_img in res_imgs:
            if s_img['file_name'] == r_img['file_name']:
                hash_[s_img['id']] = r_img['id']
                break

    return hash_

def get_id_to_anns(coco):
    annotations = coco['annotations'].copy()
    id_to_ann = dict()
    for ann in annotations:
        exist_ids = list(id_to_ann.keys())
        if ann['image_id'] in exist_ids:
            id_to_ann[ann['image_id']].append(ann)
        else:
            id_to_ann[ann['image_id']] = [ann]

    return id_to_ann


def res_to_data(res, confs, ious, detector):

    data = []
    n_confs = res.shape[0]
    n_ious = res.shape[1]
    for i in range(n_confs):
        for j in range(n_ious):
            data.append({'score': confs[i],
                         'iou': ious[j],
                         'precision': res[i, j, 0],
                         'recall': res[i, j, 1],
                         'f1': res[i, j, 2],
                         'detector': detector})

    return  data

if __name__ == "__main__":
    src_path = "/media/matans/My Book/FruitSpec/detectors_eval/val_set/instances_val.json"
    res_path = "/media/matans/My Book/FruitSpec/detectors_eval/val_set/val2017_yolox_results.json"

    res_mat, ious, confs = analyze(src_path, res_path)
    avg_res_yolox = np.nanmean(res_mat, axis=2)
    data = res_to_data(avg_res_yolox, confs, ious, 'yolox')

    print(f"Done analyzing {res_path.split('/')[-1]}")

    res_path = "/media/matans/My Book/FruitSpec/detectors_eval/val_set/val2017_yolov8_results.json"

    res_mat, ious, confs = analyze(src_path, res_path)
    avg_res_yolov8 = np.nanmean(res_mat, axis=2)
    data += res_to_data(avg_res_yolov8, confs, ious, 'yolov8')

    print(f"Done analyzing {res_path.split('/')[-1]}")

    df = pd.DataFrame(data, columns=['score', 'iou', 'precision', 'recall', 'f1', 'detector'])

    iou = 0.5
    sdf = df.query(f"iou < {iou + 0.01} and iou > {iou - 0.01}")

    print(avg_res_yolov8)