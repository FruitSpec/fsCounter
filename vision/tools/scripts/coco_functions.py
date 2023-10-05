import json
import os
from vision.misc.help_func import validate_output_path

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
    data['categories'] = [{"id": single_class_id, "name": "single_class"}]

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





if __name__ == "__main__":

    #########################################################################
    path_to_json = r'/home/lihi/FruitSpec/Data/counter/Tomato_FSI_train_260923/annotations/val_coco.json'
    # change category_id to 0 for all annotations in a COCO annotations file
    modify_coco_category_id_to_0('/home/lihi/FruitSpec/Data/counter/Apples_train_041023/annotations/instances_val.json',
                            '/home/lihi/FruitSpec/Data/counter/Apples_train_041023/annotations/instances_val_new.json')
    #######################################################################
    # Call the function
    path_to_json = r'/home/lihi/FruitSpec/Data/counter/Tomato_FSI_train_260923/annotations/val_coco.json'
    new_file_path = convert_to_single_class(path_to_json)
    print(f"Modified annotations saved to: {new_file_path}")

    ########################################################################
    # Count objects in COCO annotations:
    annotations_file_path = "/home/lihi/FruitSpec/Data/counter/Tomato_FSI_train_260923/annotations/train_coco.json"
    result = count_objects_in_coco(annotations_file_path)
    print(result)

    ######################################################################
    # Modify category based on type_position:
    coco_path = "/home/lihi/FruitSpec/Data/counter/Tomato_FSI_train_260923/all_jsons/batch9_frames_h.json"
    output_path = "/home/lihi/FruitSpec/Data/counter/Tomato_FSI_train_260923/all_jsons/corrected_jsons/batch9_frames_h.json"
    validate_output_path(os.path.dirname(output_path))
    modify_coco_category_based_on_type_position(coco_path, output_path)
    print("Done!")

    with open(output_path, "r") as file:
        content = file.readlines()

    content_sample = content[:10]

    print('done')