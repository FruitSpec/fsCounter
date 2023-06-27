import os
import fiftyone as fo
import fiftyone.utils.coco as fouc
import datetime
from tqdm import tqdm
import numpy as np

from vision.data.COCO_utils import load_coco_file


def view_coco_file(file_dict, data_path, classes=['fruit'], dataset_name=None):

    names = list(file_dict.keys())

    if dataset_name is None:
        dataset_name = create_name()

    d = fo.Dataset.from_dir(
        name=dataset_name,
        dataset_type=fo.types.COCODetectionDataset,
        data_path=data_path,
        labels_path=file_dict[names[0]],
        include_id=True
    )

    for name in names[1:]:
        fouc.add_coco_labels(d, label_field=name, labels_or_path=file_dict[name], classes=classes)

    fo.launch_app(d)

def vizualize_coco_results(file_dict, data_path, img_size=[1536, 2048], labels=['fruit'], dataset_name=None):

    if dataset_name is None:
        dataset_name = create_name()

    ann_dict = {}
    for k, v in file_dict.items():
        print(f'creating {k} hash')
        coco = load_coco_file(v)
        ann_dict[k] = create_hash(coco)

    samples = []
    print(f'Generating {dataset_name} dataset')
    for image in tqdm(coco['images']):
        sample = fo.Sample(filepath=os.path.join(data_path, image['file_name']))
        for k, hash_ in ann_dict.items():
            id_list = list(hash_.keys())
            if image['id'] in id_list:
                annotations = hash_[image['id']]
            else:
                continue
            # Convert detections to FiftyOne format
            detections = []
            for obj in annotations:
                label = labels[0]#labels[int(obj['category_id'])]
                # Bounding box coordinates should be relative values
                # in [0, 1] in the following format:
                # [top-left-x, top-left-y, width, height]
                bounding_box = obj["bbox"]
                bounding_box[0] = bounding_box[0] / img_size[0]
                bounding_box[1] = bounding_box[1] / img_size[1]
                bounding_box[2] = bounding_box[2] / img_size[0]
                bounding_box[3] = bounding_box[3] / img_size[1]

                if 'score' in list(obj.keys()):
                    confidence = np.round(obj["score"], 2)

                    detections.append(
                        fo.Detection(label=label, bounding_box=bounding_box, confidence=confidence)
                    )
                else:
                    detections.append(
                        fo.Detection(label=label, bounding_box=bounding_box)
                    )

            # Store detections in a field name of your choice
            sample[k] = fo.Detections(detections=detections)

        samples.append(sample)

    # Create dataset
    dataset = fo.Dataset(dataset_name)
    dataset.add_samples(samples)

    fo.launch_app(dataset)



def create_hash(coco):

    ann_mapping = {}
    ann_keys = []
    for ann in tqdm(coco['annotations']):
        if ann['image_id'] in ann_keys:
            ann_mapping[ann['image_id']].append(ann)
        else:
            ann_mapping[ann['image_id']] = [ann]
            ann_keys = list(ann_mapping.keys())

    return ann_mapping

def create_name():

    c_time = datetime.datetime.now()
    name = 'd' + str(c_time.year) + str(c_time.month) + str(c_time.day) + \
           '_h' + str(c_time.hour) + '_m' + str(c_time.minute) + '_s' + str(c_time.second)

    return name


if __name__ == '__main__':
    data_path = "/home/fruitspec-lab-3/FruitSpec/Data/Counter/val2017"
    files = {"GT": "/home/fruitspec-lab-3/FruitSpec/Data/Counter/val_coco.json"} #,
             #"yolox": "/home/fruitspec-lab/FruitSpec/Sandbox/yolox_tiny_hires_1024X1024/instances_res3.json",
             #'yoloV5': "/home/fruitspec-lab/FruitSpec/Data/JAI_FSI_V6_COCO/coco_resV5.json"}
    vizualize_coco_results(files, data_path, dataset_name="apples_val_2")

