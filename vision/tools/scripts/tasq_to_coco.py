import os
from tqdm import tqdm
import numpy as np
import shutil
from vision.tools.scripts.adjust_roboflow import align_iamges
from vision.misc.help_func import validate_output_path
from vision.data.COCO_utils import load_coco_file, write_coco_file, create_category_dict


def aggraegate_coco_files(folder, output_folder, categories=['fruit'], ver=1):
    files = os.listdir(folder)
    img_id = 0
    ann_id = 0
    images = []
    annotations = []

    #cat = create_category_dict(categories)
    for file in tqdm(files):
        cur_coco = load_coco_file(os.path.join(folder, file))
        cur_images = cur_coco['images'].copy()
        cur_ann = cur_coco['annotations'].copy()
        old_img_id_to_new = {}

        for image in cur_images:
            old_img_id_to_new[image['id']] = img_id
            new_image = {"id": img_id,
                     "license": 1,
                     "file_name": image['file_name'],
                     "height": image['height'],
                     "width": image['width']
                     }
            img_id += 1
            images.append(new_image)

        for ann in cur_ann:
            new_ann = {"id": ann_id,
                       "image_id": old_img_id_to_new[ann['image_id']],
                       "category_id": ann["category_id"],
                       "bbox": ann["bbox"],
                       "area": ann['area'],
                       "segmentation": [],
                       "iscrowd": 0}
            ann_id += 1
            annotations.append(new_ann)
        cat = cur_coco['categories']
    info = {"year": 2022,
            "version": ver,
            "description": "FruitSpec data from tasq",
            "contributor": "",
            "date_created": ""}

    new_coco = {'info': info, 'categories': cat, 'images': images, 'annotations': annotations}
    write_coco_file(new_coco, os.path.join(output_folder, 'coco.json'))

def split_to_train_val(coco_fp, images_folder, output_folder, val_size=0.1):

    orig_cc = load_coco_file(coco_fp)

    orig_imgs = orig_cc['images'].copy()
    orig_anns = orig_cc['annotations'].copy()
    num_of_images = len(orig_imgs)

    tot_ids = np.arange(0, num_of_images, 1)
    train_size = np.round((1 - val_size) * num_of_images).astype(np.uint16)

    train_ids = []
    for i in range(train_size):

        id_ = np.random.randint(0, len(tot_ids) - 1)
        train_ids.append(tot_ids[id_])
        tot_ids = np.delete(tot_ids, id_)

    train_images = []
    val_images = []
    for i, image in enumerate(orig_imgs):
        if i in train_ids:
            train_images.append(image)
        else:
            val_images.append(image)

    train_images, train_anns = create_subset(train_images, orig_anns)
    val_images, val_anns = create_subset(val_images, orig_anns)

    train_coco = orig_cc.copy()
    train_coco['images'] = train_images
    train_coco['annotations'] = train_anns

    copy_images(train_images, images_folder, os.path.join(output_folder, 'train2017'))

    val_coco = orig_cc.copy()
    val_coco['images'] = val_images
    val_coco['annotations'] = val_anns

    copy_images(val_images, images_folder, os.path.join(output_folder, 'val2017'))

    write_coco_file(train_coco, os.path.join(output_folder, 'train_coco.json'))
    write_coco_file(val_coco, os.path.join(output_folder, 'val_coco.json'))

def create_subset(subset_images, orig_anns):

    old_img_id_to_new = {}
    updated_subset_images = []
    img_id = 0
    for image in subset_images:
        old_img_id_to_new[image['id']] = img_id
        t_image = image.copy()
        t_image['id'] = img_id
        updated_subset_images.append(t_image)
        img_id += 1

    orig_imgs_keys = list(old_img_id_to_new.keys())
    subset_ann = []
    ann_ids = 0
    for ann in orig_anns:
        if ann['image_id'] in orig_imgs_keys:
            t_ann = ann.copy()
            t_ann['image_id'] = old_img_id_to_new[t_ann['image_id']]
            t_ann['id'] = ann_ids
            subset_ann.append(t_ann)
            ann_ids += 1

    return updated_subset_images, subset_ann


def copy_images(coco_images, input_folder, output_folder):

    validate_output_path(output_folder)
    for image in tqdm(coco_images):
        file_name = image['file_name']
        img_input = os.path.join(input_folder, file_name)
        img_output = os.path.join(output_folder, file_name)
        shutil.copy(img_input, img_output)





if __name__ == "__main__":
    folder = "/home/fruitspec-lab-3/FruitSpec/Data/Syngenta/tasq_data_270223/json"
    output_folder = "/home/fruitspec-lab-3/FruitSpec/Data/Syngenta/tasq_data_270223"
    categories = ['fruit']
    ver = 1

    aggraegate_coco_files(folder, output_folder, categories, ver)

    coco_fp = "/home/fruitspec-lab-3/FruitSpec/Data/Syngenta/tasq_data_270223/coco.json"
    images_folder = "/home/fruitspec-lab-3/FruitSpec/Data/Syngenta/tasq_data_270223/images"
    output_folder = "/home/fruitspec-lab-3/FruitSpec/Data/Syngenta/tasq_data_270223"
    split_to_train_val(coco_fp, images_folder, output_folder, val_size=0.1)

    expected_dims = [1920, 1080]
    rotation = 'clockwise'
    align_iamges("/home/fruitspec-lab-3/FruitSpec/Data/Syngenta/tasq_data_270223/train2017", expected_dims, rotation)
    align_iamges("/home/fruitspec-lab-3/FruitSpec/Data/Syngenta/tasq_data_270223/val2017", expected_dims, rotation)








