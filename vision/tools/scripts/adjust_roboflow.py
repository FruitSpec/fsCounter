import os
from vision.data.COCO_utils import load_coco_file, write_coco_file


def roboflow_to_coco(dataset_folder):
    folder_content = os.listdir(dataset_folder)
    train_folder = os.path.join(dataset_folder, 'train2017')
    val_folder = os.path.join(dataset_folder, 'val2017')

    for value in folder_content:
        if 'train' in value:
            os.rename(os.path.join(dataset_folder, value), train_folder)
        elif 'val' in value:
            os.rename(os.path.join(dataset_folder, value), val_folder)

    train_json = get_json_path(train_folder)
    val_json = get_json_path(val_folder)

    train_coco = load_coco_file(train_json)
    val_coco = load_coco_file(val_json)
    train_cat = train_coco['categories']

    new_cat = []
    class_ids = 0
    for cat in train_cat:
        if cat['supercategory'] == 'none':
            continue
        cat['id'] = class_ids
        class_ids +=1
        new_cat.append(cat)

    train_coco['categories'] = new_cat
    val_coco['categories'] = new_cat

    train_ann = train_coco['annotations'].copy()
    val_ann = val_coco['annotations'].copy()

    for ann in train_ann:
        if ann['category_id'] == 0:
            raise ValueError('got id 0')
        else:
            ann['category_id'] = ann['category_id'] - 1

    train_coco['annotations'] = train_ann

    for ann in val_ann:
        if ann['category_id'] == 0:
            raise ValueError('got id 0')
        else:
            ann['category_id'] = ann['category_id'] - 1

    val_coco['annotations'] = val_ann

    annotations_folder = os.path.join(dataset_folder, 'annotations')
    os.mkdir(annotations_folder)
    write_coco_file(train_coco, os.path.join(annotations_folder, 'instances_train.json'))
    write_coco_file(val_coco, os.path.join(annotations_folder, 'instances_val.json'))
    os.remove(train_json)
    os.remove(val_json)


def get_json_path(folder):
    json_path = None
    file_list = os.listdir(folder)

    for file in file_list:
        suffix_ = file.split('.')[-1]
        if 'json' in suffix_:
            json_path = os.path.join(folder, file)

    return json_path


if __name__ == "__main__":
    dataset_folder = '/home/yotam/FruitSpec/Data/VEG_JAI_v2i_coco'
    roboflow_to_coco(dataset_folder)