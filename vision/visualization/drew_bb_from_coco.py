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
                label = f"{cat_name[0]} {round(ann.get('score', ''),2)}"
                cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color_map[cat_id],
                            line_width)

        if save:
            validate_output_path(output_dir)
            output_path = os.path.join(output_dir, image_info['file_name'])
            cv2.imwrite(output_path, image)
            print (f'Saved: {output_path}')

        else:
            cv2.imshow('Annotated Image', image)
            cv2.waitKey(0)

    cv2.destroyAllWindows()

if __name__ == "__main__":

    coco_file_path = '/home/lihi/FruitSpec/Data/CLAHE_FSI/Tagging_Pipeline_Outputs/test_annotations/coco_annotations.json'
    image_directory = '/home/lihi/FruitSpec/Data/CLAHE_FSI/Tagging_Pipeline_Outputs/test_images'
    #image_directory = os.path.join(os.path.dirname(coco_file_path), 'all_images')
    # output_directory = os.path.join( os.path.dirname(coco_file_path) ,'frames_annotations')
    output_directory = '/home/lihi/FruitSpec/Data/CLAHE_FSI/Tagging_Pipeline_Outputs/tagged_images'

    display_coco_bboxes(coco_file_path, image_directory, output_dir=output_directory, save=True)

