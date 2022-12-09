import os
import sys

import torch
from omegaconf import OmegaConf
import cv2
from tqdm import tqdm

from vision.misc.help_func import get_repo_dir, scale_dets

repo_dir = get_repo_dir()
sys.path.append(os.path.join(repo_dir, 'vision', 'detector', 'yolo_x'))

from vision.pipelines.detection_flow import counter_detection
from vision.pipelines.run_args import make_parser
from vision.detector.yolo_x.yolox.data import COCODataset
from vision.detector.yolo_x.yolox.utils import xyxy2xywh
from vision.data.results_collector import ResultsCollector, scale
from vision.data import COCO_utils




def run(cfg, args):

    detector = counter_detection(cfg)
    results_collector = ResultsCollector()

    data_dir = os.path.join(args.data_dir, args.ds_name)
    img_list = os.listdir(data_dir)
    gt = COCO_utils.load_coco_file(args.coco_gt)
    map_ = map_img_to_id(gt)

    results = []
    for img_name in tqdm(img_list):

        id_ = map_[img_name]
        results_collector.collect_file_name(img_name)
        results_collector.collect_id(id_)

        img = cv2.imread(os.path.join(data_dir, img_name))

        output = detector.detect(img)

        scale_ = scale(detector.input_size, img.shape)
        det_outputs = scale_dets(output, scale_)
        results_collector.collect_detections(det_outputs, id_)
        try:
            results += output_to_coco(output,
                                      img.shape[0],
                                      img.shape[1],
                                      detector.input_size[0],
                                      detector.input_size[1],
                                      id_)
        except:
            a = 1

    results_collector.draw_and_save_dir(os.path.join(args.data_dir, args.ds_name), args.output_folder)
    coco_results = gt.copy()
    coco_results['annotations'] = results

    return coco_results




def output_to_coco(outputs, img_h, img_w, infer_h, infer_w, img_id):
    data_list = []
    for output in outputs:
        if output is None:
            return
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        scale = min(
            infer_h / float(img_h), infer_w / float(img_w)
        )
        bboxes /= scale
        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        bboxes = xyxy2xywh(bboxes)

        for ind in range(bboxes.shape[0]):
            pred_data = {
                "image_id": int(img_id),
                "category_id": int(cls[ind]),
                "bbox": bboxes[ind].numpy().tolist(),
                "score": scores[ind].numpy().item(),
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


def get_evaluation_dataloader(args , img_size, preprocess=None):

    evaldataset = COCODataset(
        data_dir=args.data_dir,
        json_file=args.coco_gt,
        name=args.ds_name,
        img_size=img_size,
        preproc=preprocess,
        batch_size=args.eval_batch
    )

    dataloader_kwargs = {
        "num_workers": 8,
        "pin_memory": True,
        "batch_size": args.eval_batch
    }

    eval_loader = torch.utils.data.DataLoader(evaldataset, **dataloader_kwargs)

    return eval_loader


if __name__ == "__main__":

    repo_dir = get_repo_dir()
    config_file = "/vision/pipelines/config/pipeline_config.yaml"
    #config_file = "/config/pipeline_config.yaml"
    cfg = OmegaConf.load(repo_dir + config_file)


    args = make_parser()

    args.data_dir = "/home/fruitspec-lab/FruitSpec/Data/JAI_FSI_V6x_COCO_with_zoom"
    args.coco_gt = "/home/fruitspec-lab/FruitSpec/Data/JAI_FSI_V6x_COCO_with_zoom/annotations/instances_val.json"
    args.ds_name = "val2017"
    args.eval_batch = 1
    args.output_path = '/home/fruitspec-lab/FruitSpec/Sandbox/Run_3_5_oct_2022'

    coco_results = run(cfg, args)

    COCO_utils.write_coco_file(coco_results, '/home/fruitspec-lab/FruitSpec/Sandbox/Run_3_5_oct_2022/instances_res.json')
