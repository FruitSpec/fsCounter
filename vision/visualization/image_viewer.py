import fiftyone as fo
import pandas as pd
import datetime
import os


def view_by_coco_file(data_path, labels_path, dataset_name):

    dataset_name = create_name(dataset_name)
    # Import the dataset
    d = fo.Dataset.from_dir(
        name=dataset_name,
        dataset_type=fo.types.COCODetectionDataset,
        data_path=data_path,
        labels_path=labels_path,
    )
    print(d)
    fo.launch_app(d)

def view_by_query(df: pd.DataFrame, q, parent_folder=None):

    to_view = df.query(q)
    to_view = list(to_view['img'])

    dataset_name = create_name()
    d = fo.Dataset(dataset_name)

    for img in to_view:
        if parent_folder is not None:
            img = parent_folder + img[1:]
        sample = fo.Sample(filepath=img)
        d.add_sample(sample)

    fo.launch_app(d)

def create_name(name):

    c_time = datetime.datetime.now()
    suffix = 'd' + str(c_time.year) + str(c_time.month) + str(c_time.day) + \
           '_h' + str(c_time.hour) + '_m' + str(c_time.minute) + '_s' + str(c_time.second)

    if name is not None:
        name = name + suffix
    else:
        name = suffix
    return name

if __name__ == "__main__":

    # df = pd.read_csv(csv_path)
    # parent_folder = path
    # view_by_query(df, 'score > 0.4', parent_folder)

    group = 'test'
    dataset_name = 'VIDEO FSI.v1i.coco'
    data_path = '/home/fruitspec-lab/FruitSpec/Data/VIDEO_FSI'
    labels_path = os.path.join(data_path, dataset_name, 'annotations', f'instances_{group}.json')
    view_by_coco_file(os.path.join(data_path, dataset_name, group), labels_path, dataset_name)