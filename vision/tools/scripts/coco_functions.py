import json
import os
from vision.misc.help_func import validate_output_path
import pickle
from tqdm import tqdm
import shutil
import random
from datetime import datetime
import matplotlib.pyplot as plt


def count_objects_in_coco(annotations_file_path):
    """
    Count the number of objects of each class in a COCO annotations file and provide additional dataset statistics.

    :param annotations_file_path: Path to the COCO annotations file.
    :return: Dictionary with class names as keys and counts as values, total number of images,
             average objects per image (rounded to nearest integer), and maximum objects in an image.
    """
    with open(annotations_file_path, 'r') as f:
        data = json.load(f)

    # Extract annotations, categories, and images
    annotations = data['annotations']
    categories = data['categories']
    images = data['images']

    # Create a mapping from category ID to category name
    cat_id_to_name = {cat['id']: cat['name'] for cat in categories}

    # Count objects for each category ID
    cat_id_counts = {}
    for annotation in annotations:
        cat_id = annotation['category_id']
        cat_id_counts[cat_id] = cat_id_counts.get(cat_id, 0) + 1

    # Convert category ID counts to category name counts
    cat_name_counts = {cat_id_to_name[cat_id]: count for cat_id, count in cat_id_counts.items()}

    # Compute additional statistics
    total_images = len(images)

    # Count objects per image
    objects_per_image = {}
    for annotation in annotations:
        image_id = annotation['image_id']
        objects_per_image[image_id] = objects_per_image.get(image_id, 0) + 1

    avg_objects_per_image = round(sum(objects_per_image.values()) / total_images)
    max_objects_in_image = max(objects_per_image.values())

    return {
        'class_counts': cat_name_counts,
        'total_images': total_images,
        'avg_objects_per_image': avg_objects_per_image,
        'max_objects_in_image': max_objects_in_image
    }



def filter_coco_annotations(input_path, output_path, category_id_to_keep, new_category_id):
    """
    Filters the COCO annotations to keep only the bounding boxes with a specific category_id
    and changes that category_id to a new one.

    Parameters:
    - input_path (str): The file path for the input COCO annotations.
    - output_path (str): The file path to save the filtered COCO annotations.
    - category_id_to_keep (int): The original category_id of the bounding boxes to keep.
    - new_category_id (int): The new category_id to assign to the filtered bounding boxes.

    Returns:
    - str: The file path to the saved filtered COCO annotations.
    """

    # Load the COCO annotations from the input file
    with open(input_path, 'r') as file:
        coco_data = json.load(file)

    # Filter out annotations that don't match the category_id_to_keep
    filtered_annotations = [anno for anno in coco_data['annotations']
                            if anno['category_id'] == category_id_to_keep]

    # Update the category_id in the annotations to the new category_id
    for anno in filtered_annotations:
        anno['category_id'] = new_category_id

    # Update the annotations in the COCO data
    coco_data['annotations'] = filtered_annotations

    # Filter the categories to keep only the one that matches category_id_to_keep
    # and update its id to the new_category_id
    filtered_categories = [category for category in coco_data['categories']
                           if category['id'] == category_id_to_keep]
    for category in filtered_categories:
        category['id'] = new_category_id

    # Update the categories in the COCO data
    coco_data['categories'] = filtered_categories

    # Save the updated COCO data to the output file
    with open(output_path, 'w') as file:
        json.dump(coco_data, file, indent=4)

    return output_path





def screen_coco_file(input_path, output_path):

    def calculate_intersection_over_smaller_area(box_a, box_b):
        """
        Calculate the intersection area as a percentage of the smaller bounding box's area.
        """
        from shapely.geometry import box

        a = box(box_a[0], box_a[1], box_a[0] + box_a[2], box_a[1] + box_a[3])
        b = box(box_b[0], box_b[1], box_b[0] + box_b[2], box_b[1] + box_b[3])
        intersection_area = a.intersection(b).area
        smaller_area = min(a.area, b.area)
        return intersection_area / smaller_area if smaller_area != 0 else 0


    with open(input_path, 'r') as file:
        data = json.load(file)

    annotations = data['annotations']
    to_remove = set()

    for i in tqdm(range(len(annotations)), desc="Processing Annotations"):
        ann_a = annotations[i]
        for j in range(len(annotations)):
            if i != j and ann_a['image_id'] == annotations[j]['image_id']:
                ann_b = annotations[j]
                intersection_ratio = calculate_intersection_over_smaller_area(ann_a['bbox'], ann_b['bbox'])
                if intersection_ratio > 0.9:
                    # Remove the larger box
                    area_a = ann_a['bbox'][2] * ann_a['bbox'][3]
                    area_b = ann_b['bbox'][2] * ann_b['bbox'][3]
                    larger_ann = i if area_a > area_b else j
                    to_remove.add(larger_ann)

    # Remove the identified annotations
    filtered_annotations = [ann for i, ann in enumerate(annotations) if i not in to_remove]
    data['annotations'] = filtered_annotations

    with open(output_path, 'w') as file:
        json.dump(data, file)

def grounding_dino_to_coco(pkl_file_path, output_json_path):
    # Load the pkl file
    with open(pkl_file_path, 'rb') as file:
        dino_results = pickle.load(file)

    # Prepare the COCO format dictionary
    coco_format = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    category_set = set()
    annotation_id = 1

    for image_name, data in dino_results.items():
        # Add image info
        image_id = len(coco_format["images"]) + 1
        coco_format["images"].append({
            "id": image_id,
            "width": data["size"][1],
            "height": data["size"][0],
            "file_name": image_name  # Include image name
        })

        # Convert and add each bbox
        for box, label in zip(data["boxes"], data["labels"]):
            # Convert tensor to list and from [x_center, y_center, width, height] to [x, y, width, height]
            box = box.tolist()
            x_center, y_center, width, height = box
            x = (x_center - width / 2) * data["size"][1]  # De-normalize and convert to top-left x
            y = (y_center - height / 2) * data["size"][0]  # De-normalize and convert to top-left y
            norm_width = width * data["size"][1]  # De-normalized width
            norm_height = height * data["size"][0]  # De-normalized height
            area = norm_width * norm_height  # Area of the bbox

            # Extract label and confidence
            label, confidence = label.split('(')
            confidence = float(confidence[:-1])  # Remove the closing parenthesis and convert to float

            # Check if the category already exists
            if label not in category_set:
                category_set.add(label)
                coco_format["categories"].append({
                    "id": len(category_set) - 1,  # Subtract 1 here
                    "name": label
                })

            # Find the category ID
            category_id = next((category["id"] for category in coco_format["categories"] if category["name"] == label), None)

            # Add annotation
            coco_format["annotations"].append({
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [x, y, norm_width, norm_height],
                "area": area,  # Include the area of the bbox
                "score": confidence
            })

            annotation_id += 1

    # Write the COCO format data to a JSON file
    with open(output_json_path, 'w') as json_file:
        json.dump(coco_format, json_file, indent=4)

    return output_json_path

def remove_absent_images_from_coco(coco_file_path, images_dir_path, output_coco_file_path):
    # Load the COCO file
    with open(coco_file_path, 'r') as file:
        coco_data = json.load(file)

    # Get the set of image filenames in the images directory
    present_images = set(os.listdir(images_dir_path))

    # Filter out images not present in the directory
    filtered_images = [img for img in coco_data['images'] if img['file_name'] in present_images]

    # Update the COCO data
    coco_data['images'] = filtered_images

    # Save the updated COCO file
    with open(output_coco_file_path, 'w') as file:
        json.dump(coco_data, file)

    print (f'Saved: {output_coco_file_path}')

class COCOInspector:
    def __init__(self, coco_file_path: str, images_dir: str, output_dir: str):
        self.coco_file_path = coco_file_path
        self.output_dir = output_dir
        self.annotations = None
        self.images_dir = images_dir
        self._load_annotations()

    def _load_annotations(self):
        """
        Load annotations from the COCO file.
        """
        with open(self.coco_file_path, "r") as file:
            self.annotations = json.load(file)


    def has_type_position(self) -> bool:
        """
        Check if 'type_position' is present in the annotations.
        """
        return any('type_position' in annotation for annotation in self.annotations['annotations'])

    def modify_coco_category_based_on_type_position(self):
        """
        Modify the category_id in COCO annotations based on the type_position value.
        """
        # Checking if 'type_position' is present
        if not self.has_type_position():
            print("No 'type_position' field found.")
            return

        # Update the category_id in annotations based on type_position
        for annotation in self.annotations['annotations']:
            if annotation['type_position'] == 'A':
                annotation['category_id'] = 0
            elif annotation['type_position'] == 'B':
                annotation['category_id'] = 1

        # Update the categories list
        self.annotations['categories'] = [
            {"id": 0, "name": "A"},
            {"id": 1, "name": "B"}
        ]


    def convert_all_category_ids_to_0(self):
        '''
        Change the category_id of all annotations to 0 in a COCO annotations file.
        '''

        # Modify the categories list
        self.annotations["categories"] = [{"id": 0, "name": "fruit"}]

        # Change the category_id of all annotations to 0
        for annotation in self.annotations["annotations"]:
            annotation["category_id"] = 0
        print(f'Chenged all categories_ids to 0')

    def save(self):
        """
        Save the annotations in a new file with a timestamp.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_parts = self.coco_file_path.split('.')
        new_filename = f"{'.'.join(filename_parts[:-1])}_{timestamp}.{filename_parts[-1]}"

        with open(new_filename, "w") as file:
            json.dump(self.annotations, file, indent=4)

        print (f'Saved: {new_filename}')

        return new_filename

    def _handle_previous_data(self, current_coco, path_previous_coco, images_dir_previous, images_dir_current):
        """
        Add data from previous COCO files.
        """
        with open(path_previous_coco, 'r') as f:
            previous_coco = json.load(f)

        for file in os.listdir(images_dir_previous):
            if file.endswith('.jpg'):
                shutil.copy(os.path.join(images_dir_previous, file), images_dir_current)
                print(f'Copied: {images_dir_current}/{file}')
        print(f'Copied all images from {images_dir_previous} to {images_dir_current}')

        current_coco['annotations'] += previous_coco['annotations']
        current_coco['images'] += previous_coco['images']

        return current_coco

    def move_images(self, image_list, dest_dir):
        """
        Move images from the specified source directory to the destination directory.
        """
        for img in image_list:
            source_file = os.path.join(self.images_dir, img['file_name'])
            dest_file = os.path.join(dest_dir, img['file_name'])
            if os.path.exists(source_file):
                shutil.move(source_file, dest_file)
                print(f'Moved {img["file_name"]} to {dest_dir}')
            else:
                print(f"Image not found: {source_file}")

    def count_annotations(self):
        """
        Counts the number of images and annotations for each class, and the total number of annotations.
        """
        class_counts = {}
        image_counts = len(self.annotations['images'])
        total_annotations = 0

        for annotation in self.annotations['annotations']:
            category_id = annotation['category_id']
            class_counts[category_id] = class_counts.get(category_id, 0) + 1
            total_annotations += 1

        # Print the results
        print(f"Total number of images: {image_counts}")
        print(f"Total number of annotations: {total_annotations}")
        for category in self.annotations['categories']:
            category_id = category['id']
            category_name = category['name']
            count = class_counts.get(category_id, 0)
            print(f"Class '{category_name}' (ID: {category_id}): {count} annotations")


    def train_test_split(self, split_ratio=0.85, seed=42, path_previous_coco_train=None,
                         path_previous_coco_test=None, path_previous_images_train=None,
                         path_previous_images_test=None):
        """
        Splits the data into training and testing sets based on the provided split ratio.
        """
        random.seed(seed)
        random.shuffle(self.annotations['images'])

        split_index = int(len(self.annotations['images']) * split_ratio)

        train_images_list = self.annotations['images'][:split_index]
        test_images_list = self.annotations['images'][split_index:]

        train_annotations = [ann for ann in self.annotations['annotations'] if
                             ann['image_id'] in [img['id'] for img in train_images_list]]
        test_annotations = [ann for ann in self.annotations['annotations'] if
                            ann['image_id'] in [img['id'] for img in test_images_list]]

        train_coco = {
            "images": train_images_list,
            "annotations": train_annotations,
            "categories": self.annotations['categories']
        }
        test_coco = {
            "images": test_images_list,
            "annotations": test_annotations,
            "categories": self.annotations['categories']
        }

        train_images_dir = os.path.join(self.output_dir, 'train2017')
        test_images_dir = os.path.join(self.output_dir, 'val2017')
        os.makedirs(train_images_dir, exist_ok=True)
        os.makedirs(test_images_dir, exist_ok=True)

        self.move_images(train_images_list, train_images_dir)
        self.move_images(test_images_list, test_images_dir)

        if path_previous_coco_train and path_previous_coco_test and path_previous_images_train and path_previous_images_test:
            train_coco = self._handle_previous_data(train_coco, path_previous_coco_train, path_previous_images_train,
                                                    train_images_dir)
            test_coco = self._handle_previous_data(test_coco, path_previous_coco_test, path_previous_images_test,
                                                   test_images_dir)

        today_date = datetime.now().strftime("%d%m%Y")
        train_coco_path = os.path.join(self.output_dir, 'annotations', f'coco_train_{today_date}.json')
        test_coco_path = os.path.join(self.output_dir, 'annotations', f'coco_test_{today_date}.json')

        self.save_coco_json(train_coco, train_coco_path)
        self.save_coco_json(test_coco, test_coco_path)

        return train_coco_path, test_coco_path

    def save_coco_json(self, data, file_path): #todo remove
        """
        Save COCO formatted data to a JSON file.
        """
        validate_output_path(file_path)

        with open(file_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Saved COCO file: {file_path}")


def plot_bbox_area_histogram(coco_path):
    """
    Plots a histogram of the areas of bounding boxes from a COCO dataset file.

    Parameters:
    coco_path (str): Path to the COCO dataset file.
    """
    # Load the COCO dataset
    with open(coco_path, 'r') as file:
        coco_data = json.load(file)

    # Extract bounding box areas
    bbox_areas = [ann['bbox'][2] * ann['bbox'][3] for ann in coco_data['annotations']]

    # Plotting the histogram
    plt.figure(figsize=(10, 6))
    plt.hist(bbox_areas, bins=10, color='blue', alpha=0.7)
    plt.title('Histogram of Bounding Box Areas')
    plt.xlabel('Area')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    coco_file_path = '/home/fruitspec-lab-3/FruitSpec/Data/Counter/CLAHE_FSI/D240102_01-Citrus_FSI_CLAHE_batch12_14_WITHOUT_APPLES/annotations/coco_train_31122023.json'
    images_dir_path = '/home/fruitspec-lab-3/FruitSpec/Data/Counter/CLAHE_FSI/D240102_01-Citrus_FSI_CLAHE_batch12_14_WITHOUT_APPLES/train2017'
    output_coco_file_path = '/home/fruitspec-lab-3/FruitSpec/Data/Counter/CLAHE_FSI/D240102_01-Citrus_FSI_CLAHE_batch12_14_WITHOUT_APPLES/annotations/coco_train_31122023_WITHOUT_APPLES.json'
    remove_absent_images_from_coco(coco_file_path, images_dir_path, output_coco_file_path)
    #########################################33

    #plot_bbox_area_histogram('/home/fruitspec-lab-3/FruitSpec/Data/Counter/CLAHE_FSI/D231231_01-Citrus_FSI_CLAHE_batch12_14/annotations/batch14.json')
    ############################################################
    # coco_file_path = '/home/fruitspec-lab-3/FruitSpec/Data/Counter/CLAHE_FSI/batch14/export_batch14.json'
    #
    # coco_inspector = COCOInspector(coco_file_path, images_dir = '', output_dir= '')
    # coco_inspector.count_annotations()
    # coco_inspector.convert_all_category_ids_to_0()
    # coco_inspector.count_annotations()
    # coco_inspector.save()
    ################################################################33

    coco_file_path = r'/home/fruitspec-lab-3/FruitSpec/Data/Counter/CLAHE_FSI/batch14/export_batch14_20231231_230732.json'
    images_dir = r'/home/fruitspec-lab-3/FruitSpec/Data/Counter/CLAHE_FSI/batch14/all_images'
    output_dir = r'/home/fruitspec-lab-3/FruitSpec/Data/Counter/CLAHE_FSI/batch14'

    path_previous_coco_train = "/home/fruitspec-lab-3/FruitSpec/Data/Counter/CLAHE_FSI/D231218_01_batch12_Citrus_CLAHE/annotations/train_coco.json"
    path_previous_coco_test = "/home/fruitspec-lab-3/FruitSpec/Data/Counter/CLAHE_FSI/D231218_01_batch12_Citrus_CLAHE/annotations/val_coco.json"
    path_previous_images_train = "/home/fruitspec-lab-3/FruitSpec/Data/Counter/CLAHE_FSI/D231218_01_batch12_Citrus_CLAHE/train2017"
    path_previous_images_test = "/home/fruitspec-lab-3/FruitSpec/Data/Counter/CLAHE_FSI/D231218_01_batch12_Citrus_CLAHE/val2017"


    coco_inspector = COCOInspector(coco_file_path, images_dir, output_dir)
    coco_inspector.count_annotations()
    # coco_inspector.convert_all_category_ids_to_0()
    # coco_inspector.has_type_position()
    # coco_inspector.modify_coco_category_based_on_type_position()
    # coco_inspector.save()

    train_coco_path, test_coco_path = coco_inspector.train_test_split(
        split_ratio=0.85,
        path_previous_coco_train=path_previous_coco_train,
        path_previous_coco_test=path_previous_coco_test,
        path_previous_images_train=path_previous_images_train,
        path_previous_images_test=path_previous_images_test
    )

    ##################################################################################
    images_dir_path = r'/home/fruitspec-lab-3/FruitSpec/Data/Counter/CLAHE_FSI/batch12/all_images'
    coco_file_path = r'/home/fruitspec-lab-3/FruitSpec/Data/Counter/CLAHE_FSI/batch12/batch_12_class0.json'
    output_coco_file_path = r'/home/fruitspec-lab-3/FruitSpec/Data/Counter/CLAHE_FSI/batch12/batch_12_class0_reduced.json'
    remove_absent_images_from_coco(coco_file_path, images_dir_path, output_coco_file_path)
#########################################################################
    coco_file_path = r'/home/fruitspec-lab-3/FruitSpec/Data/Counter/CLAHE_FSI/batch12/batch_12.json'
    modify_coco_category_id_to_0(coco_file_path)
########################################################################
    # coco_file_path = '/home/fruitspec-lab-3/FruitSpec/Data/Counter/syngenta/FSI/annotations/train_coco.json'
    # convert_to_single_class(coco_file_path)


#########################################################################################################
    # Define the input and output paths
    input_file_path = '/home/fruitspec-lab-3/FruitSpec/Data/Counter/syngenta/coco_all_2_classes.json'
    output_file_path = '/home/fruitspec-lab-3/FruitSpec/Data/Counter/syngenta/coco_1_classe_whole.json'


    # Call the function to filter the annotations
    output_path = filter_coco_annotations(
        input_file_path,
        output_file_path,
        category_id_to_keep=50,
        new_category_id=0)

    #########################################################################
    path_to_json = r'/home/fruitspec-lab-3/FruitSpec/Data/Counter/Apples_train_051023/annotations/val_coco.json'
    # change category_id to 0 for all annotations in a COCO annotations file
    modify_coco_category_id_to_0(path_to_json)
    # #######################################################################
    # # Call the function
    # path_to_json = r'/home/lihi/FruitSpec/Data/counter/Tomato_FSI_train_260923/annotations/val_coco.json'
    # new_file_path = convert_to_single_class(path_to_json)
    # print(f"Modified annotations saved to: {new_file_path}")
    #
    # ########################################################################
    # # Count objects in COCO annotations:
    # annotations_file_path = "/home/lihi/FruitSpec/Data/counter/Tomato_FSI_train_260923/annotations/train_coco.json"
    # result = count_objects_in_coco(annotations_file_path)
    # print(result)
    #
    # ######################################################################
    # # Modify category based on type_position:
    # coco_path = "/home/lihi/FruitSpec/Data/counter/Tomato_FSI_train_260923/all_jsons/batch9_frames_h.json"
    # output_path = "/home/lihi/FruitSpec/Data/counter/Tomato_FSI_train_260923/all_jsons/corrected_jsons/batch9_frames_h.json"
    # validate_output_path(os.path.dirname(output_path))
    # modify_coco_category_based_on_type_position(coco_path, output_path)
    # print("Done!")
    #
    # with open(output_path, "r") as file:
    #     content = file.readlines()
    #
    # content_sample = content[:10]

    print('done')