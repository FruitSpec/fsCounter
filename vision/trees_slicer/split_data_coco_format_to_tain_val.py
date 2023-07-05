import os
import random
import shutil
import json

import os
import random
import shutil
import json

'''
Split a set of images and their corresponding Coco annotations into train and validation sets. 
It organizes the train and validation images in separate directories and saves the train and validation annotation 
files in an annotations directory. This can be useful for preparing data for YOLOX custom training or similar tasks.
'''
def split_train_val_images(all_images_dir, coco_annotation_file, train_ratio = 0.8):
    # Load Coco annotations from JSON file
    with open(coco_annotation_file, 'r') as f:
        coco_annotations = json.load(f)

    # Create directories for train and validation images
    base_dir = os.path.dirname(all_images_dir)
    train_images_dir = os.path.join(base_dir, 'train2017')
    val_images_dir = os.path.join(base_dir, 'val2017')
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)

    # Shuffle the images and annotations
    random.shuffle(coco_annotations['images'])

    # Split the images into train and validation sets
    num_train_images = int(len(coco_annotations['images']) * train_ratio)
    train_images = coco_annotations['images'][:num_train_images]
    val_images = coco_annotations['images'][num_train_images:]

    # Copy train images to the train directory
    for image in train_images:
        image_path = os.path.join(all_images_dir, image['file_name'])
        shutil.copy(image_path, os.path.join(train_images_dir, image['file_name']))

    # Copy validation images to the validation directory
    for image in val_images:
        image_path = os.path.join(all_images_dir, image['file_name'])
        shutil.copy(image_path, os.path.join(val_images_dir, image['file_name']))

    # Create directory for annotations
    annotations_dir = os.path.join(base_dir, 'annotations')
    os.makedirs(annotations_dir, exist_ok=True)

    # Save train and validation annotation files
    train_annotations = {
        'images': train_images,
        'annotations': [ann for ann in coco_annotations['annotations'] if ann['image_id'] in [img['id'] for img in train_images]],
        'categories': coco_annotations['categories']
    }
    val_annotations = {
        'images': val_images,
        'annotations': [ann for ann in coco_annotations['annotations'] if ann['image_id'] in [img['id'] for img in val_images]],
        'categories': coco_annotations['categories']
    }
    train_annotations_file = os.path.join(annotations_dir, 'instances_train.json')
    val_annotations_file = os.path.join(annotations_dir, 'instances_val.json')
    with open(train_annotations_file, 'w') as f:
        json.dump(train_annotations, f)
    with open(val_annotations_file, 'w') as f:
        json.dump(val_annotations, f)

    print("Train and validation images and annotations are split and copied successfully.")


# # Example usage
# all_images_dir = '/home/lihi/FruitSpec/Data/training_yoloX/slicer_data_rgd/all_images'
# coco_annotation_file = '/home/lihi/FruitSpec/Data/training_yoloX/slicer_data_rgd/annotations/all_annotations.json'

all_images_dir = '/home/fruitspec-lab-3/FruitSpec/Data/customers/DEWAGD/training_yoloX/slicer_data_rgd/all_images'
coco_annotation_file = '/home/fruitspec-lab-3/FruitSpec/Data/customers/DEWAGD/training_yoloX/slicer_data_rgd/annotations/all_annotations.json'


split_train_val_images(all_images_dir, coco_annotation_file, train_ratio=0.8)
