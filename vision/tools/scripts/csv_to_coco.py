import pandas as pd

from vision.data.COCO_utils import load_coco_file, write_coco_file

def convert(template, csv_path, output_path, expected_dims=[2048, 1536]):

    df = pd.read_csv(csv_path)
    csv_list = []
    for row, v in df.iterrows():
        csv_list.append(v.to_list())


    key_to_img = {}
    for k, v in enumerate(csv_list):
        f_id = int(v[-1])
        if f_id in key_to_img.keys():
            key_to_img[f_id].append(k)
        else:
            key_to_img[f_id] = [k]

    images = []
    for img_id in key_to_img.keys():
        new_image = {"id": img_id,
                     "license": 1,
                     "file_name": f"frame_{img_id}.jpg",
                     "height": expected_dims[0],
                     "width": expected_dims[1]
                     }
        images.append(new_image)

    annotations = []
    for img_id, ann_id in key_to_img.items():
        for ann in ann_id:
            bbox = csv_list[ann][:4]
            bbox[2] = bbox[2] - bbox[0] # width
            bbox[3] = bbox[3] - bbox[1]  # hight
            area = (bbox[3]*bbox[2])

            new_ann = {"id": ann,
                       "image_id": img_id,
                       "category_id": 0,
                       "bbox": bbox,
                       "area": area,
                       "segmentation": [],
                       "iscrowd": 0}
            annotations.append(new_ann)

    coco = load_coco_file(template)
    coco['annotations'] = annotations
    coco['images'] = images

    write_coco_file(coco, output_path)

if __name__ == "__main__":

    template = "/home/fruitspec-lab/FruitSpec/Data/JAI_FSI_V6_COCO/annotations/instances_val.json"
    output_path = "/home/fruitspec-lab/FruitSpec/Sandbox/Counter/clahe_test/EQUALIZE_HIST_2/coco_det.json"
    csv_path = "/home/fruitspec-lab/FruitSpec/Sandbox/Counter/clahe_test/EQUALIZE_HIST_2/detections.csv"

    convert(template, csv_path, output_path)

    output_path = "/home/fruitspec-lab/FruitSpec/Sandbox/Counter/clahe_test/clahe_2/coco_det.json"
    csv_path = "/home/fruitspec-lab/FruitSpec/Sandbox/Counter/clahe_test/clahe_2/detections.csv"

    convert(template, csv_path, output_path)
