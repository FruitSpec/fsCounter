import os
import json
import shutil

import numpy as np


def convert_to_coco_format(outputs, class_ids, type_='dets', data_label=None):
    data_list = []

    # preprocessing: resize
    # scale = min(img_size[0] / float(info_imgs[0]), img_size[1] / float(info_imgs[1]))
    for output in outputs:
        if output is None:
            continue
        bboxes = output[0:4]
        # bboxes /= scale
        scores = output[4]
        cls = output[5]
        track_id = output[6]
        ids = output[7]

        if type_ == 'tracks' and track_id == -1:
            continue

        bboxes = xyxy2xywh(bboxes)

        label = class_ids[int(cls)]
        pred_data = {
            "image_id": int(ids),
            "category_id": int(label),
            "bbox": bboxes,
            "score": float(scores),
            "segmentation": [],
            "track_id": int(track_id)
        }  # COCO json format
        if data_label is not None:
            pred_data["label"] = data_label
        data_list.append(pred_data)

    return data_list


def create_category_dict(category_list):
    categories = []
    for i, category in enumerate(category_list):
        categories.append({"id": i,
                           "name": category
                           })
    return categories


def create_images_dict(files, ids, height, width):
    images = []
    for file, id_ in zip(files, ids):
        image = {"id": int(id_),
                 "license": 1,
                 "file_name": file,
                 "height": int(height),
                 "width": int(width),
                 }
        images.append(image)

    return images


def xyxy2xywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    return bboxes

def generate_coco_format(outputs, info_imgs, ids, img_size,
                         class_ids, category_list, file_list, type_="dets", data_label=None):

    coco = dict()

    coco["categories"] = create_category_dict(category_list)
    coco["images"] = create_images_dict(file_list, ids, info_imgs[0], info_imgs[1])
    coco["annotations"] = convert_to_coco_format(outputs, class_ids, type_, data_label)

    return coco


def write_coco_file(coco_data, output_path):
    with open(output_path, "w") as f:
        json.dump(coco_data, f)


def load_coco_file(file_path):
    with open(file_path, "r") as f:
        coco = json.load(f)

    return coco


def split_without_fx(coco_dir, out_dir, train_size=0.8, val_size=0.2):
    f_coco = accumulate_without_fx(coco_dir)
    train_dir, val_dir, test_dir, annotations_dir = get_coco_folders(coco_dir)
    split_and_copy(f_coco, out_dir, train_dir, val_dir, test_dir, train_size, val_size)

def accumulate_without_fx(coco_dir):

    train_dir, val_dir, test_dir, annotations_dir = get_coco_folders(coco_dir)
    train_annotations, val_annotations, test_annotations = get_annotations_files(annotations_dir)

    t_coco = load_coco_file(os.path.join(annotations_dir, train_annotations))
    t_images, t_annotations, img_id, ann_id = keep_jai_data(t_coco)

    v_coco = load_coco_file(os.path.join(annotations_dir, val_annotations))
    v_images, v_annotations, img_id, ann_id = keep_jai_data(v_coco, img_id, ann_id)

    if not(test_dir is None):
        te_coco = load_coco_file(os.path.join(annotations_dir, test_annotations))
        te_images, te_annotations, img_id, ann_id = keep_jai_data(te_coco, img_id, ann_id)

        f_coco_images = t_images + v_images + te_images
        f_coco_ann = t_annotations + v_annotations + te_annotations

    else:
        f_coco_images = t_images + v_images
        f_coco_ann = t_annotations + v_annotations


    f_coco = {'info': t_coco['info'],
              'licenses': t_coco['licenses'],
              'categories': t_coco['categories'],
              'images': f_coco_images,
              'annotations': f_coco_ann}

    return f_coco


def split_and_copy(f_coco, out_dir, train_dir, val_dir, test_dir=None, train_size=0.8, val_size=0.2):
    images = f_coco['images']
    annotations = f_coco['annotations']

    tot_images = len(images)

    n_train_imgs = int(tot_images * train_size)

    if test_dir is None:  # no test set
        n_val_imgs = tot_images - n_train_imgs
    else:
        n_val_imgs = int(tot_images * val_size)
        n_test_imgs = tot_images - n_train_imgs - n_val_imgs

    tot_index_list = list(range(0, tot_images))

    ann_dir = os.path.join(out_dir, 'annotations')
    os.mkdir(ann_dir)

    # split train images
    train_images, train_ann, tot_index_list = copy_subset(images, annotations, out_dir, 'train', n_train_imgs,
                                                          tot_index_list, train_dir, val_dir, test_dir)
    train_coco = {'info': f_coco['info'],
                  'licenses': f_coco['licenses'],
                  'categories': f_coco['categories'],
                  'images': train_images,
                  'annotations': train_ann}

    write_coco_file(train_coco, os.path.join(ann_dir, 'instances_train.json'))

    # split val images
    val_images, val_ann, tot_index_list = copy_subset(images, annotations, out_dir, 'val', n_val_imgs,
                                                      tot_index_list, train_dir, val_dir, test_dir)

    val_coco = {'info': f_coco['info'],
                  'licenses': f_coco['licenses'],
                  'categories': f_coco['categories'],
                  'images': val_images,
                  'annotations': val_ann}

    write_coco_file(val_coco, os.path.join(ann_dir, 'instances_val.json'))

    if test_dir:
        test_images, test_ann, tot_index_list = copy_subset(images, annotations, out_dir, 'test', n_test_imgs,
                                                          tot_index_list, train_dir, val_dir, test_dir)

        test_coco = {'info': f_coco['info'],
                    'licenses': f_coco['licenses'],
                    'categories': f_coco['categories'],
                    'images': test_images,
                    'annotations': test_ann}

        write_coco_file(val_coco, os.path.join(ann_dir, 'instances_test.json'))


def copy_subset(images, annotations, out_dir, subset_name, n_imgs, tot_index_list, train_dir, val_dir, test_dir):

    count = 0
    ids = []
    subset_images = []
    out_dir = os.path.join(out_dir, subset_name)
    os.mkdir(out_dir)
    while count < n_imgs:
        cur_ind = np.random.randint(0, len(tot_index_list))
        cur_img_ind = tot_index_list[cur_ind]

        cur_image = images[cur_img_ind]
        image_name = cur_image['file_name']
        image_id = cur_image['id']

        # copy
        copy_from = get_full_img_path(image_name, train_dir, val_dir, test_dir)
        copy_to = os.path.join(out_dir, image_name)
        shutil.copy(copy_from, copy_to)

        ids.append(image_id)
        subset_images.append(cur_image)

        # remove from tot_index_list
        tot_index_list.remove(tot_index_list[cur_ind])

        count += 1

    subset_ann = []
    for ann in annotations:
        if ann['image_id'] in ids:
            subset_ann.append(ann)

    return subset_images, subset_ann, tot_index_list


def get_full_img_path(image_name, train_dir, val_dir, test_dir=None):
    orig_train_file_list = os.listdir(train_dir)
    orig_val_file_list = os.listdir(val_dir)
    if test_dir:
        orig_test_file_list = os.listdir(test_dir)

    if image_name in orig_train_file_list:
        copy_from = os.path.join(train_dir, image_name)
    elif image_name in orig_val_file_list:
        copy_from = os.path.join(val_dir, image_name)
    elif not (orig_test_file_list is None):
        if image_name in orig_test_file_list:
            copy_from = os.path.join(test_dir, image_name)
    else:
        print(f'File {image_name} not found')
        copy_from = None
    return copy_from



def get_coco_folders(coco_dir):
    folder_list = os.listdir(coco_dir)
    test_dir = None  # not always test is existing
    for item_ in folder_list:
        if 'train' in item_ and os.path.isdir(os.path.join(coco_dir, item_)):
            train_dir = os.path.join(coco_dir, item_)
        elif 'val' in item_ and os.path.isdir(os.path.join(coco_dir, item_)):
            val_dir = os.path.join(coco_dir, item_)
        elif 'test' in item_ and os.path.isdir(os.path.join(coco_dir, item_)):
            test_dir = os.path.join(coco_dir, item_)
        elif 'annotations' in item_ and os.path.isdir(os.path.join(coco_dir, item_)):
            annotations_dir = os.path.join(coco_dir, item_)

    return train_dir, val_dir, test_dir, annotations_dir


def get_annotations_files(annotations_dir):
    f_list = os.listdir(annotations_dir)
    test_annotations = None  # not always test is existing
    for f in f_list:
        if os.path.isdir(os.path.join(annotations_dir, f)):
            continue
        name, suffix = f.split('.')
        if 'json' in suffix:
            if 'train' in name:
                train_annotations = f
            elif 'val' in name:
                val_annotations = f
            elif 'test' in name:
                test_annotations = f

    return train_annotations, val_annotations, test_annotations


def keep_jai_data(coco, img_id=0, ann_id=0):
    """
    remove FX images from dataset

    FX images obtained by the str 'RADIANCE' or 'row'
    """

    images = coco['images']
    annotations = coco['annotations']

    new_image = []
    old_image_ids = []
    new_annotations = []

    for image in images:
        # dont copy images from FX
        if 'RADIANCE' in image['file_name'] or 'row' in image['file_name']:
            continue
        else:
            new_image.append(image)
            old_image_ids.append(image['id'])

    for annotation in annotations:
        if annotation['image_id'] in old_image_ids:
            new_annotations.append(annotation)
    final_image = []
    final_ann = []
    for image in new_image:
        used_ann = []
        for annotation in new_annotations:
            if annotation['image_id'] == image['id']:
                annotation['image_id'] = img_id
                annotation['id'] = ann_id
                used_ann.append(annotation)
                final_ann.append(annotation)
                ann_id += 1
        for ann in used_ann:
            new_annotations.remove(ann)
        image['id'] = img_id
        final_image.append(image)
        img_id += 1

    return final_image, final_ann, img_id, ann_id


def create_hash(coco):
    ann_mapping = dict()
    ann_keys = list()
    for ann in coco['annotations']:
        if ann['image_id'] in ann_keys:
            ann_mapping[ann['image_id']].append(ann.copy())
        else:
            ann_mapping[ann['image_id']] = [ann.copy()]
            ann_keys = list(ann_mapping.keys())

    return ann_mapping


if __name__ == "__main__":
    out_dir = '/home/fruitspec-lab/FruitSpec/Data/JAI_FSI_V1_COCO'
    coco_dir = '/home/fruitspec-lab/FruitSpec/Data/FX_JAI_FSI_V1_COCO/COCO'
    split_without_fx(coco_dir, out_dir, train_size=0.8, val_size=0.2)








