import os
import cv2
import json

'''
The code iterates over a folder of images with the space key, 
and displays bbox from coco json annotation file.
'''
def display_bounding_boxes(image_path, annotations_path, target_width):
    # Load Coco annotations from JSON file
    with open(annotations_path, 'r') as f:
        coco_annotations = json.load(f)

    # Find the image ID based on the image filename
    image_filename = os.path.basename(image_path)
    image_id = next((image['id'] for image in coco_annotations['images'] if image['file_name'] == image_filename), None)
    if image_id is None:
        print(f"Image {image_path} not found in the annotations.")
        return

    # Find annotations for the specific image
    image_annotations = [ann for ann in coco_annotations['annotations'] if ann['image_id'] == image_id]

    # Load the image with OpenCV
    image = cv2.imread(image_path)

    # Resize the image while maintaining aspect ratio
    aspect_ratio = image.shape[1] / image.shape[0]
    target_height = int(target_width / aspect_ratio)
    image_resized = cv2.resize(image, (target_width, target_height))

    # Create a copy of the resized image to draw bounding boxes on
    image_with_boxes = image_resized.copy()

    # Adjust the bounding box coordinates to match the resized image dimensions
    scale_x = target_width / image.shape[1]
    scale_y = target_height / image.shape[0]

    # Add bounding boxes to the image
    for annotation in image_annotations:
        bbox = annotation['bbox']
        x, y, width, height = bbox
        x_resized = int(x * scale_x)
        y_resized = int(y * scale_y)
        width_resized = int(width * scale_x)
        height_resized = int(height * scale_y)
        cv2.rectangle(image_with_boxes, (x_resized, y_resized), (x_resized + width_resized, y_resized + height_resized), (0, 255, 0), 2)

    # Display the resized image with bounding boxes using OpenCV
    cv2.imshow('Image with Bounding Boxes', image_with_boxes)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("Image:", image_filename)
def iterate_images(folder_path, annotations_file, target_width):
    # Iterate over images in the folder
    for file_name in os.listdir(folder_path):
        if file_name.endswith('.jpg') or file_name.endswith('.jpeg') or file_name.endswith('.png'):
            image_path = os.path.join(folder_path, file_name)
            display_bounding_boxes(image_path, annotations_file, target_width)


# Example usage
folder_path = '/home/lihi/FruitSpec/Data/slicer_data_rgd/all_images'
annotations_file = '/home/lihi/FruitSpec/Data/slicer_data_rgd/annotations/all_annotations.json'
iterate_images(folder_path, annotations_file,500)
