import json
import os
from vision.misc.help_func import validate_output_path
import json
import pickle
#from shapely.geometry import box
from tqdm import tqdm



def modify_coco_category_based_on_type_position(original_path: str, output_path: str) -> str:
    """
    For use when having coco annotations that the classes are defined by 'type_position' field instead of 'category_id'.
    Modify the category_id in COCO annotations based on the type_position value.

    Parameters:
    - original_path: Path to the original COCO annotations file.
    - output_path: Path to save the modified annotations.

    Returns:
    - Path to the modified annotations.
    """
    # Load the annotations from the provided file
    with open(original_path, "r") as file:
        annotations = json.load(file)

    # Update the category_id in annotations based on type_position
    for annotation in annotations['annotations']:
        if annotation['type_position'] == 'A':
            annotation['category_id'] = 0
        elif annotation['type_position'] == 'B':
            annotation['category_id'] = 1

    # Update the categories list
    annotations['categories'] = [
        {"id": 0, "name": "A"},
        {"id": 1, "name": "B"}
    ]

    # Save the modified annotations to the specified output path with indentation
    with open(output_path, "w") as file:
        json.dump(annotations, file, indent=4)

    return output_path


def modify_coco_category_id_to_0(coco_file_path):
    '''
    Change the category_id of all annotations to 0 in a COCO annotations file.
    '''

    # Load the COCO annotations JSON file
    with open(coco_file_path, 'r') as f:
        data = json.load(f)

    # Modify the categories list
    data["categories"] = [{"id": 0, "name": "fruit"}]

    # Change the category_id of all annotations to 0
    for annotation in data["annotations"]:
        annotation["category_id"] = 0

    # Define the path to save the modified annotations
    base_name = os.path.basename(coco_file_path).split('.')[0]
    dir_name = os.path.dirname(coco_file_path)
    new_file_name = f"{base_name}_class0.json"
    new_file_path = os.path.join(dir_name, new_file_name)

    # Save the modified data to a new JSON file
    with open(new_file_path, 'w') as f:
        json.dump(data, f, indent=4)

    print(f'Saved: {new_file_path}')


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

def convert_to_single_class(coco_file_path):
    """
    Convert all classes in a COCO annotations file to a single class and save the modified annotations.

    :param coco_file_path: Path to the original COCO annotations file.
    """
    # Load the original COCO annotations
    with open(coco_file_path, 'r') as f:
        data = json.load(f)

    # Modify the categories to contain only a single class
    single_class_id = 1
    data['categories'] = [{"id": single_class_id, "name": "Fruit"}]

    # Update the category_id in the annotations to reference the new single class ID
    for annotation in data['annotations']:
        annotation['category_id'] = single_class_id

    # Define the path to save the modified annotations
    base_name = os.path.basename(coco_file_path).split('.')[0]
    dir_name = os.path.dirname(coco_file_path)
    new_file_name = f"{base_name}_single_class.json"
    new_file_path = os.path.join(dir_name, new_file_name)

    # Save the modified COCO annotations
    with open(new_file_path, 'w') as f:
        json.dump(data, f, indent=4)

    return new_file_path





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



def calculate_intersection_over_smaller_area(box_a, box_b):
    """
    Calculate the intersection area as a percentage of the smaller bounding box's area.
    """
    a = box(box_a[0], box_a[1], box_a[0] + box_a[2], box_a[1] + box_a[3])
    b = box(box_b[0], box_b[1], box_b[0] + box_b[2], box_b[1] + box_b[3])
    intersection_area = a.intersection(b).area
    smaller_area = min(a.area, b.area)
    return intersection_area / smaller_area if smaller_area != 0 else 0

def screen_coco_file(input_path, output_path):
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

# Example usage

if __name__ == "__main__":
    images_dir_path = r'/home/fruitspec-lab-3/FruitSpec/Data/Counter/CLAHE_FSI/batch12/all_images'
    coco_file_path = r'/home/fruitspec-lab-3/FruitSpec/Data/Counter/CLAHE_FSI/batch12/batch_12_class0.json'
    output_coco_file_path = r'/home/fruitspec-lab-3/FruitSpec/Data/Counter/CLAHE_FSI/batch12/batch_12_class0_reduced.json'
    remove_absent_images_from_coco(coco_file_path, images_dir_path, output_coco_file_path)
#########################################################################
    coco_file_path = r'/home/fruitspec-lab-3/FruitSpec/Data/Counter/CLAHE_FSI/batch12/batch_12.json'
    modify_coco_category_id_to_0(coco_file_path)
########################################################################
    coco_file_path = '/home/fruitspec-lab-3/FruitSpec/Data/Counter/syngenta/FSI/annotations/train_coco.json'
    convert_to_single_class(coco_file_path)


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