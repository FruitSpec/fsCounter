import os
import cv2
import numpy as np

from vision.data.COCO_utils import load_coco_file, write_coco_file, create_hash

def zoom_in(image, annotations, threshold, desired_ann_width):
    """ create zoomed-in images for images with bbox width above threshold"""
    orig_h = image.shape[0]
    orig_w = image.shape[1]
    for ann in annotations:
        if (ann['bbox'][2] > threshold) and (ann['bbox'][2] < desired_ann_width):  # width
            r = desired_ann_width / ann['bbox'][2]
            a = ann.copy()
            break
    bbox = a['bbox']

    # max possible box movement
    hor_front = int(orig_w - bbox[0] - bbox[2])
    hor_back = int(bbox[0])
    ver_front = int(orig_h - bbox[1] - bbox[3])
    ver_back = int(bbox[1])


    # random horizontal shift direction
    crop_w = orig_w / r
    hor_shift = np.random.randint(0, min(hor_back, hor_front) + 1)
    if hor_front < hor_back:
        hor_range = [int(orig_w - crop_w - hor_shift), int(orig_w - hor_shift)]
    else:
        hor_range = [int(hor_shift), int(hor_shift + crop_w)]

    crop_h = orig_h / r
    ver_shift = np.random.randint(0, min(ver_back, ver_front) + 1)
    if ver_front < ver_back:
        ver_range = [int(orig_h - crop_h - ver_shift), int(orig_h - ver_shift)]
    else:
        ver_range = [int(ver_shift), int(ver_shift + crop_h)]
    if min(hor_range) < 0 or min(ver_range) < 0:
        return None, []
    cropped = image[ver_range[0]: ver_range[1], hor_range[0]: hor_range[1], :]
    cropped_image = cv2.resize(cropped, (orig_w, orig_h))



    shifted_ann = []
    for a in annotations:
        bbox = a['bbox'].copy()
        bbox[0] = bbox[0] - hor_range[0]
        bbox[1] = bbox[1] - ver_range[0]

        if bbox[0] < 0 and bbox[0] + bbox[2] <= crop_w and bbox[0] + bbox[2] > 0:
            bbox[2] -= np.abs(bbox[0])
            bbox[0] = 0
        elif bbox[0] >= 0 and bbox[0] < crop_w and  bbox[0] + bbox[2] > crop_w:
            bbox[2] -= (bbox[0] + bbox[2] - crop_w)

        if bbox[1] < 0 and bbox[1] + bbox[3] <= crop_h and bbox[1] + bbox[3] > 0:
            bbox[3] -= np.abs(bbox[1])
            bbox[1] = 0
        elif bbox[1] >= 0 and bbox[1] < crop_h and bbox[1] + bbox[3] > crop_h:
            bbox[3] -= (bbox[1] + bbox[3] - crop_h)

        if bbox[0] >= 0 and bbox[0] <= crop_w:
            if bbox[1] >= 0 and bbox[1] <= crop_h:
                bbox = list(np.array(bbox) * r)
                #bbox = [int(b) for b in bbox]
                a['bbox'] = bbox
                shifted_ann.append(a)

    return cropped_image, shifted_ann



if __name__ == "__main__":
    coco_fp = ['/home/fruitspec-lab/FruitSpec/Data/JAI_FSI_V6_COCO/annotations/instances_train.json',
               '/home/fruitspec-lab/FruitSpec/Data/JAI_FSI_V6_COCO/annotations/instances_val.json']
    data_dir = ['/home/fruitspec-lab/FruitSpec/Data/JAI_FSI_V6_COCO/train2017',
                '/home/fruitspec-lab/FruitSpec/Data/JAI_FSI_V6_COCO/val2017']

    output_dir = ['/home/fruitspec-lab/FruitSpec/Data/JAI_FSI_V6x_COCO_with_zoom/train2017',
                '/home/fruitspec-lab/FruitSpec/Data/JAI_FSI_V6x_COCO_with_zoom/val2017']

    output_json = ['/home/fruitspec-lab/FruitSpec/Data/JAI_FSI_V6x_COCO_with_zoom/annotations/instances_train.json',
                   '/home/fruitspec-lab/FruitSpec/Data/JAI_FSI_V6x_COCO_with_zoom/annotations/instances_val.json']

    ids = [0, 1]
    for id_ in ids:
        coco = load_coco_file(coco_fp[id_])
        images = coco['images'].copy()
        ann = coco['annotations'].copy()
        h = create_hash(coco.copy())

        last_im_id = len(images)
        ann_id = len(ann)
        image_ids = []
        for a in ann:
            if a['bbox'][2] > 80:
                if a['image_id'] not in image_ids:
                    image_ids.append(a['image_id'])


        for img_id in image_ids:
            desired_ann_width = np.random.randint(150, 251)
            image_name = images[img_id]['file_name']
            im = cv2.imread(os.path.join(data_dir[id_], image_name))

            dets = h[img_id]
            c_img, im_ann = zoom_in(im, dets, 80, desired_ann_width)
            if c_img is None:
                continue
            f_name = image_name.split('.')[0] + '_zm.jpg'
            cv2.imwrite(os.path.join(output_dir[id_], f_name), c_img)

            cur_coco_image = {'id': last_im_id,
                              'license': 1,
                              'file_name': f_name,
                              'height': 2048,
                              'width': 1536,
                              'date_captured': '2022-09-19T17:10:28+00:00'}

            images.append(cur_coco_image)

            for im_a in im_ann:
                im_a['id'] = ann_id
                im_a['image_id'] = last_im_id
                ann_id += 1

            ann += im_ann

            last_im_id += 1

        coco['images'] = images
        coco['annotations'] = ann

        write_coco_file(coco, output_json[id_])




