from collections import defaultdict
import json

def convert_to_coco_format(outputs, info_imgs, ids, img_size, class_ids, data_label=None, return_outputs=False):
    data_list = []
    image_wise_data = defaultdict(dict)
    for (output, img_h, img_w, img_id) in zip(
            outputs, info_imgs[0], info_imgs[1], ids
    ):
        if output is None:
            continue
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        scale = min(
            img_size[0] / float(img_h), img_size[1] / float(img_w)
        )
        bboxes /= scale
        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]
        tracks = output[:, 7]

        image_wise_data.update({
            int(img_id): {
                "bboxes": [box.numpy().tolist() for box in bboxes],
                "scores": [score.numpy().item() for score in scores],
                "categories": [
                    class_ids[int(cls[ind])]
                    for ind in range(bboxes.shape[0])
                ],
            }
        })

        bboxes = xyxy2xywh(bboxes)

        for ind in range(bboxes.shape[0]):
            label = class_ids[int(cls[ind])]
            pred_data = {
                "image_id": int(img_id),
                "category_id": label,
                "bbox": bboxes[ind].numpy().tolist(),
                "score": scores[ind].numpy().item(),
                "segmentation": [],
                "track_id": tracks[ind]
            }  # COCO json format
            if data_label is not None:
                pred_data["label"] = data_label
            data_list.append(pred_data)

    if return_outputs:
        return data_list, image_wise_data
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
                         class_ids, category_list, file_list, data_label=None):

    coco = dict()

    coco["categories"] = create_category_dict(category_list)
    coco["images"] = create_images_dict(file_list, ids, info_imgs[0], info_imgs[1])
    coco["annotations"] = convert_to_coco_format(outputs, info_imgs, ids,
                                                 img_size, class_ids, data_label)

    return coco


def write_coco_file(coco_data, output_path):
    with open(output_path, "w") as f:
        json.dump(coco_data, f)