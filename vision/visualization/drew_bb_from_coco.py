import json
import os
import cv2
from vision.misc.help_func import validate_output_path

def display_coco_bboxes(coco_file, img_dir, max_height=1000, line_width=1, output_dir=None, save=True):
    # Load COCO annotations
    with open(coco_file, 'r') as f:
        annotations = json.load(f)

    # Define a list of distinct colors
    all_colors = [
        (0, 0, 255), (0, 255, 0), (255, 0, 0), (0, 255, 255), (255, 255, 0),
        (255, 0, 255), (127, 127, 255), (127, 255, 127), (255, 127, 127)
    ]
    num_classes = len(annotations['categories'])
    colors = all_colors * (num_classes // len(all_colors)) + all_colors[:num_classes % len(all_colors)]

    color_map = {cat['id']: colors[i] for i, cat in enumerate(annotations['categories'])}

    # Font scale relative to line width
    font_scale = line_width / 2.0

    # Go through each image in the dataset
    for image_info in annotations['images']:
        image_path = os.path.join(img_dir, image_info['file_name'])
        image = cv2.imread(image_path)

        # Resize image for display if its height is too large
        height, width = image.shape[:2]
        scale = max_height / height
        if scale < 1:
            image = cv2.resize(image, (int(scale * width), int(scale * height)))

        # Find the annotations for this image
        for ann in annotations['annotations']:
            if ann['image_id'] == image_info['id']:
                # Extract bbox and class information
                bbox = [int(val * scale) for val in ann['bbox']]
                cat_id = ann['category_id']
                cat_name = [cat['name'] for cat in annotations['categories'] if cat['id'] == cat_id][0]

                # Draw bbox and annotations
                x, y, w, h = bbox
                cv2.rectangle(image, (x, y), (x + w, y + h), color_map[cat_id], line_width)
                label = f"{cat_name[0]} {ann.get('score', '')}"
                cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_map[cat_id],
                            line_width)

        if save:
            output_path = os.path.join(output_dir, image_info['file_name'])
            cv2.imwrite(output_path, image)
            print (f'Saved: {output_path}')

        else:
            cv2.imshow('Annotated Image', image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


coco_file_path = '/home/fruitspec-lab-3/FruitSpec/Data/Counter/syngenta/FSI/annotations/train_coco_single_class.json'
image_directory = '/home/fruitspec-lab-3/FruitSpec/Data/Counter/syngenta/FSI/train2017'
output_directory = '/home/fruitspec-lab-3/FruitSpec/Data/Counter/syngenta/FSI/annotated_images'
validate_output_path(output_directory)
display_coco_bboxes(coco_file_path, image_directory, output_dir=output_directory, save=True)
