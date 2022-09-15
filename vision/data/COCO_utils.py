import json

def convert_to_coco_format(outputs, info_imgs, img_size, class_ids, type_='dets', data_label=None):
    data_list = []

    # preprocessing: resize
    scale = min(
        img_size[0] / float(info_imgs[0]), img_size[1] / float(info_imgs[1])
    )

    for output in outputs():
        if output is None:
            continue
        bboxes = output[0:4]
        bboxes /= scale
        scores = output[4] * output[5]
        cls = output[6]
        ids = output[7]
        track_id = output[8]

        if type_ == 'tracks' and track_id == -1:
            continue

        bboxes = xyxy2xywh(bboxes)

        label = class_ids[int(cls)]
        pred_data = {
            "image_id": int(ids),
            "category_id": label,
            "bbox": bboxes.numpy().tolist(),
            "score": scores.numpy().item(),
            "segmentation": [],
            "track_id": track_id
        }  # COCO json format
        if data_label is not None:
            pred_data["label"] = data_label
        data_list.append(pred_data)

    return data_list


def create_category_dict(category_list):
    categories = []
    for i, category in category_list:
        categories.append({"id": i,
                           "name": category
                           })
    return categories


def create_images_dict(files, ids, height, width):
    images = []
    for file, id_ in zip(files, ids):
        image = {"id": id_,
                 "license": 1,
                 "file_name": file,
                 "height": height,
                 "width": width,
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
    coco["annotations"] = convert_to_coco_format(outputs, info_imgs,
                                                 img_size, class_ids, type_, data_label)

    return coco


def write_coco_file(coco_data, output_path):
    with open(output_path, "w") as f:
        json.dump(coco_data, f)