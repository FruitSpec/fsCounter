import os
import json

def convert(filepath, resize_factor=3):
    mp4_wrong_height = int(1080 // resize_factor)
    wrong_r = min(mp4_wrong_height / 1080, mp4_wrong_height / 1920)
    real_height = int(1920 // resize_factor)
    real_r = min(real_height / 1080, real_height / 1920)

    data = load_json(filepath)
    converted_data = {}
    for k, v in data.items():
        temp_v = v.copy()
        if temp_v['start'] is not None:
            temp_v['start'] = int(temp_v['start'] / wrong_r * real_r)
        if temp_v['end'] is not None:
            temp_v['end'] = int(temp_v['end'] / wrong_r * real_r)
        if len(temp_v['left_clusters']) > 1:
            for count, bbox in temp_v['left_clusters'].items():
                if count == 'count':
                    continue
                bbox[0] = int(bbox[0] / wrong_r * real_r)
                bbox[1] = int(bbox[1] / wrong_r * real_r)
                bbox[2] = int(bbox[2] / wrong_r * real_r)
                bbox[3] = int(bbox[3] / wrong_r * real_r)
        if len(temp_v['right_clusters']) > 1:
            for count, bbox in temp_v['right_clusters'].items():
                if count == 'count':
                    continue
                bbox[0] = int(bbox[0] / wrong_r * real_r)
                bbox[1] = int(bbox[1] / wrong_r * real_r)
                bbox[2] = int(bbox[2] / wrong_r * real_r)
                bbox[3] = int(bbox[3] / wrong_r * real_r)

        converted_data[k] = temp_v
    write_json(filepath, converted_data)

def write_json(filepath, data):
    with open(filepath, 'w') as f:
        json.dump(data, f)

def load_json(filepath):

    with open(filepath, 'r') as f:
        loaded_data = json.load(f)
    data = {}
    for k, v in loaded_data.items():
        data[int(k)] = v

    return data

if __name__ == "__main__":

    for scan in ['/media/yotam/Extreme SSD/syngenta trail/tomato/analysis/200323/pre', '/media/yotam/Extreme SSD/syngenta trail/tomato/analysis/200323/post']:
        for row in ['7','8','9','10','11','18','19','20','21']:
            try:
                json_path = os.path.join(scan, row,
                                         [i for i in os.listdir(os.path.join(scan, row)) if 'slice_data' in i][0])
                convert(json_path)
                print(json_path)
            except:
                print('FAILED',)
                continue