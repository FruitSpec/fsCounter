import cv2
import os
from tqdm import tqdm
from vision.data.COCO_utils import load_coco_file, write_coco_file


def roboflow_to_coco(dataset_folder, expected_dims, rotation='clockwise'):
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
        class_ids += 1
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

    align_iamges(train_folder, expected_dims, rotation)
    align_iamges(val_folder, expected_dims, rotation)


def get_json_path(folder):
    json_path = None
    file_list = os.listdir(folder)

    for file in file_list:
        suffix_ = file.split('.')[-1]
        if 'json' in suffix_:
            json_path = os.path.join(folder, file)

    return json_path


def align_iamges(folder, expected_dims, rotation='clockwise'):
    rotate = cv2.ROTATE_90_CLOCKWISE if rotation == 'clockwise' else cv2.ROTATE_90_COUNTERCLOCKWISE

    file_list = os.listdir(folder)

    for file in tqdm(file_list):
        suffix = file.split('.')[-1]
        if 'jpg' in suffix or 'png' in suffix:
            img = cv2.imread(os.path.join(folder, file))
            if img.shape[0] == expected_dims[0] and img.shape[0] == expected_dims[0]:
                continue

            img = cv2.rotate(img, rotate)
            cv2.imwrite(os.path.join(folder, file), img)

def curate_ds(dataset_folder, wrong):
    annotations_folder = os.path.join(dataset_folder, 'annotations')

    train_data = load_coco_file(os.path.join(annotations_folder, 'instances_train.json'))
    train_data = remove_wrong_samples_from_coco(train_data, wrong)
    write_coco_file(train_data, os.path.join(dataset_folder, 'instances_train_curated.json'))

    val_data = load_coco_file(os.path.join(annotations_folder, 'instances_val.json'))
    val_data = remove_wrong_samples_from_coco(val_data, wrong)
    write_coco_file(val_data, os.path.join(dataset_folder, 'instances_val_curated.json'))


def remove_wrong_samples_from_coco(coco, wrong):
    images = coco['images'].copy()
    ann = coco['annotations'].copy()

    images_to_remove = []
    for img in images:
        for w in wrong:
            if w in img['file_name']:
                images_to_remove.append(img['id'])
                break

    new_train_images = []
    old_to_new_id = {}
    id_ = 0
    for img in images:
        if img['id'] in images_to_remove:
            continue
        old_to_new_id[img['id']] = id_
        c_img = img.copy()
        c_img['id'] = id_
        id_ += 1
        new_train_images.append(c_img)

    new_ann = []
    id_ = 0
    for a in ann:
        if a['image_id'] in images_to_remove:
            continue
        c_ann = a.copy()
        c_ann['image_id'] = old_to_new_id[c_ann['image_id']]
        c_ann['id'] = id_
        id_ += 1
        new_ann.append(c_ann)

    coco['images'] = new_train_images
    coco['annotations'] = new_ann

    return coco








if __name__ == "__main__":
    #dataset_folder = '/home/yotam/FruitSpec/Data/VEG_JAI_v4i_coco'
    dataset_folder = "/home/yotam/FruitSpec/Data/VEG_RGB_v6i_coco"
    expected_dims = [1920, 1080]
    rotation = 'clockwise'

    #roboflow_to_coco(dataset_folder, expected_dims, rotation)

    wrong = ['18199_12-06-05', '18199_12-09-03', '18199_12-11-32', '18199_12-13-29']
    curate_ds(dataset_folder, wrong)
    #align_iamges(os.path.join(dataset_folder, 'train2017'), expected_dims, rotation)
    #align_iamges(os.path.join(dataset_folder, 'val2017'), expected_dims, rotation)

