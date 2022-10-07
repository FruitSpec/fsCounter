import os
import pandas
import json

def load_coco_file(file_path):
    with open(file_path, 'r') as f:
        coco = json.load(f)
    return coco
