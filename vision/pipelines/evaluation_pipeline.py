import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf
import cv2
from tqdm import tqdm

from vision.misc.help_func import get_repo_dir, scale_dets

repo_dir = get_repo_dir()
sys.path.append(os.path.join(repo_dir, 'vision', 'detector', 'yolo_x'))

from vision.pipelines.detection_flow import counter_detection
from vision.detector.yolo_x.yolox.utils import xyxy2xywh
from vision.data.COCO_utils import create_images_dict, write_coco_file




def run(cfg, args, data_path, test_conf=0.01):

    cfg.detector.confidence = test_conf
    detector = counter_detection(cfg, args, False)

    img_list = os.listdir(data_path)

    id_ = 0

    annotations = []
    ids = []
    for img_name in tqdm(img_list):

        img = cv2.imread(os.path.join(data_path, img_name))

        output = detector.detect([img])

        ids.append(id_)
        if len(output) > 0:
            annotations += output_to_coco(output,
                                          id_)

        id_ += 1

    coco = dict()
    coco['images'] = create_images_dict(img_list, ids, img.shape[0], img.shape[1])
    coco['annotations'] = annotations

    return coco




def output_to_coco(outputs, img_id):
    data_list = []
    for output in outputs:
        if (output is None) or len(output) == 0:
            continue
        #output = output.cpu()
        output = np.array(output)

        bboxes = output[:, 0:4]


        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        bboxes = xyxy2xywh(bboxes)

        for ind in range(bboxes.shape[0]):
            pred_data = {
                "image_id": int(img_id),
                "category_id": int(cls[ind]),
                "bbox": bboxes[ind].tolist(),
                "score": scores[ind].item(),
                "segmentation": [],
            }  # COCO json format
            data_list.append(pred_data)

    return data_list

    # dataloader = get_evaluation_dataloader(args, img_size=cfg.input_size, preprocess=ValTransform(legacy=False))
    # evaluator = COCOEvaluator(dataloader,
    #                           img_size=cfg.input_size,
    #                           confthre=cfg.detector.confidence,
    #                           nmsthre=cfg.detector.nms,
    #                           num_classes=cfg.detector.num_of_classes)
    #
    # eval_results, data_list = evaluator.evaluate(detector.detector,
    #                                   half=cfg.detector.fp16, output_eval=True)
    #
    # return eval_results, data_list

def map_img_to_id(gt):
    mapped = dict()
    imgs = gt['images']
    for img in imgs:
        mapped[img['file_name']] = img['id']

    return mapped


# def get_evaluation_dataloader(args , img_size, preprocess=None):
#
#     evaldataset = COCODataset(
#         data_dir=args.data_dir,
#         json_file=args.coco_gt,
#         name=args.ds_name,
#         img_size=img_size,
#         preproc=preprocess,
#         batch_size=args.eval_batch
#     )
#
#     dataloader_kwargs = {
#         "num_workers": 8,
#         "pin_memory": True,
#         "batch_size": args.eval_batch
#     }
#
#     eval_loader = torch.utils.data.DataLoader(evaldataset, **dataloader_kwargs)
#
#     return eval_loader


if __name__ == "__main__":

    repo_dir = get_repo_dir()
    config_file = "/vision/pipelines/config/pipeline_config.yaml"
    runtime_config = "/vision/pipelines/config/dual_runtime_config.yaml"
    cfg = OmegaConf.load(repo_dir + config_file)
    args = OmegaConf.load(repo_dir + runtime_config)

    data_path = "/media/matans/My Book/FruitSpec/detectors_eval/val_set/val2017"
    output_path = "/media/matans/My Book/FruitSpec/detectors_eval/val_set"
    test_conf = 0.01



    coco_results = run(cfg, args, data_path, test_conf)

    splited = data_path.split('/')
    coco_name = f'{splited[-1]}_{cfg.detector.detector_type}_results.json'
    write_coco_file(coco_results, os.path.join(output_path, coco_name))
