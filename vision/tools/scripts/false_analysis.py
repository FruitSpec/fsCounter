import os

import numpy as np
from tqdm import tqdm

from vision.data.COCO_utils import load_coco_file, write_coco_file, create_hash

def analyze(gt_file_path, det_file_path, output_path, iou=0.5):

    no_match_bboxes = []
    no_match_images = []

    gt_coco = load_coco_file(gt_file_path)
    det_coco = load_coco_file(det_file_path)

    h_gt = create_hash(gt_coco)
    h_det = create_hash(det_coco)
    det_Ids = list(h_det.keys())

    gt_map = map_img_to_id(gt_coco)
    det_map = map_img_to_id(det_coco)

    for img_name, gt_id in tqdm(gt_map.items()):
        if img_name not in list(det_map.keys()):
            gt_bbox = h_gt[gt_id]
            no_match = gt_bbox
        else:
            det_id = det_map[img_name]
            gt_bbox = h_gt[gt_id]
            if det_id in det_Ids:
                det_bbox = h_det[det_id]
                # find best matches by IoU
                no_match = []
                for gt in gt_bbox:
                    scores = []
                    bb_a = gt['bbox'].copy()
                    for det in det_bbox:
                        bb_b = det['bbox'].copy()
                        scores.append(calc_iou(bb_a, bb_b))
                    if np.max(scores) < iou:
                        no_match.append(gt)
            else:
                no_match = gt_bbox

        if no_match:  # not empty
            no_match_images.append(gt_id)
            no_match_bboxes += no_match

    images = gt_coco['images'].copy()
    reduced_images = []
    for im in images:
        if im['id'] in no_match_images:
            reduced_images.append(im)


    gt_coco['images'] = reduced_images
    gt_coco['annotations'] = no_match_bboxes

    output_file_name = os.path.join(output_path, 'res.json')
    write_coco_file(gt_coco, output_file_name)



def map_img_to_id(gt):
    mapped = dict()
    imgs = gt['images']
    for img in imgs:
        mapped[img['file_name']] = img['id']

    return mapped

def calc_iou(bb_a, bb_b):

    ax = max(bb_a[0], bb_b[0])
    bx = min(bb_a[0] + bb_a[2], bb_b[0] + bb_b[2])
    ay = max(bb_a[1], bb_b[1])
    by = min(bb_a[1] + bb_a[3], bb_b[1] + bb_b[3])

    inter = max(0, bx - ax + 1) * max(0, by - ay + 1)
    area_a = bb_a[2] * bb_a[3]
    area_b = bb_b[2] * bb_b[3]

    iou = inter / float(area_a + area_b - inter)

    return iou



if __name__ == "__main__":

    det_path = '/home/fruitspec-lab/FruitSpec/Sandbox/Counter/clahe_test/clahe_2/coco_det.json'
    gt_path = "/home/fruitspec-lab/FruitSpec/Sandbox/Counter/clahe_test/EQUALIZE_HIST_2/coco_det.json"

    output_path = "/home/fruitspec-lab/FruitSpec/Sandbox/Counter/clahe_test/"

    analyze(gt_path, det_path, output_path, 0.4)
