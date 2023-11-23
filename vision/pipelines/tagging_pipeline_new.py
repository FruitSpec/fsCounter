import cv2
import pandas as pd
import json
import os
from tqdm import tqdm
from vision.tools.utils_general import find_subdirs_with_file, download_s3_files
from vision.misc.help_func import get_subpath_from_dir
import shutil
import random
from datetime import datetime

class TaggingPipeline:
    """
    A utility for processing videos and tracking data to extract frames and compile annotations into a COCO-format JSON.

    Usage:
        - Initialize with paths to video folders and output directory.
        - Run the pipeline specifying whether to save frames and/or update COCO JSON.
        - Outputs frames and a COCO JSON with annotations for object detection models.

    The video_identifier is created from the last five subdirectories of the video's path.
    """

    def __init__(self, output_dir, rotate_option=None, frames_interval=10):
        self.output_dir = output_dir
        self.rotate_option = rotate_option
        self.frames_interval = frames_interval
        self.annotation_counter = 1  # Initialize annotation ID counter

    @staticmethod
    def validate_output_path(path):
        if not os.path.exists(path):
            os.makedirs(path)

    @staticmethod
    def frames_idx(frames_amount, frames_interval):
        return [x for x in range(frames_amount) if x % frames_interval == 0]

    def rotate_frame(self, frame):
        if self.rotate_option == 'clockwise':
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotate_option == 'counter_clockwise':
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    def process_video(self, video_path, tracks_csv_path, video_identifier, save_frames, update_coco):

        self.tracking_results = pd.read_csv(tracks_csv_path)
        tot_frames = int(self.tracking_results['frame_id'].max())

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video stream or file: {video_path}")
            return

        height  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        width = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if not hasattr(self, 'frames_idx_list'): # If frames_idx_list doesn't exist, create it
            self.frames_idx_list = self.frames_idx(tot_frames, self.frames_interval)
        frame_save_path = os.path.join(self.output_dir, 'frames')
        self.validate_output_path(frame_save_path)

        if save_frames:
            for frame_id in self.frames_idx_list:
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                ret, frame = cap.read()
                if ret:
                    frame = self.rotate_frame(frame)
                    frame_file_path = os.path.join(frame_save_path, f"{video_identifier}_frame_{frame_id}.jpg")
                    success = cv2.imwrite(frame_file_path, frame)
                    if not success:
                        print(f"Failed to save frame {frame_id}")
                    else:
                        print(f"Saved {frame_file_path}")
                else:
                    print(f"Failed to capture frame {frame_id}")

        cap.release()

        # Process tracks and update COCO JSON with a progress bar
        if update_coco:

            filtered_tracks = self.tracking_results[self.tracking_results['frame_id'].isin(self.frames_idx_list)]
            self.update_coco_json(filtered_tracks, video_identifier, width=width, height=height)

    def update_coco_json(self, df, video_identifier, width, height):

        # get classes from tracs.csv file:
        class_ids = self.tracking_results['class_conf'].unique().astype(int).tolist()

        self.coco_format = {
            "videos": [],
            "images": [],
            "annotations": [],
            "categories": [
                {
                    "id": class_id,  # Use the provided class_id directly
                    "name": f"class_{class_id}",  # Name can be customized as needed
                    "supercategory": "none"
                } for class_id in class_ids  # Iterate over the provided class IDs
            ]
        }
        self.coco_format["videos"].append(video_identifier)

        for _, row in df.iterrows():
            frame_id = int(row['frame_id'])
            image_file_name = f"{video_identifier}_frame_{frame_id}.jpg"

            # Check if the image has already been added by looking up the file name
            existing_image = next(
                (image for image in self.coco_format["images"] if image['file_name'] == image_file_name), None)

            if not existing_image:
                # If the image doesn't exist, increment the image_id counter and add the image
                image_id = len(self.coco_format["images"]) + 1
                image = {
                    "id": image_id,
                    "width": width,  # Width is not available in the provided data
                    "height": height,  # Height is not available in the provided data
                    "file_name": image_file_name
                }
                self.coco_format["images"].append(image)
            else:
                # If the image exists, use the existing image_id
                image_id = existing_image["id"]

            # Add the annotation with the image_id
            annotation = {
                "id": self.annotation_counter,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [row['x1'], row['y1'], row['x2'] - row['x1'], row['y2'] - row['y1']],
                "area": (row['x2'] - row['x1']) * (row['y2'] - row['y1']),
                "iscrowd": 0,
                "segmentation": [],
                "score": row['obj_conf']
            }
            self.coco_format["annotations"].append(annotation)
            self.annotation_counter += 1  # Increment the annotation ID counter

    def save_coco_json(self, data, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=2)
        print(f'Saved: {file_path}')

    def run(self, videos_folder, save_frames=True, update_coco=True, video_name = 'Result_FSI.mkv'):

        self.videos_folder = videos_folder
        subdirs = find_subdirs_with_file(self.videos_folder, file_name = video_name, return_dirs=True, single_file=False)
        for subdir in subdirs:

            video_path = os.path.join(subdir, video_name)
            tracks_csv_path = os.path.join(subdir, 'tracks.csv')
            video_identifier = '_'.join(subdir.split(os.sep)[-6:])

            self.process_video(video_path, tracks_csv_path, video_identifier, save_frames, update_coco)

        # if update_coco:   #todo uncomment
        #     self.save_coco_json(data = self.coco_format, file_path = os.path.join(self.output_dir, 'coco_dataset.json') )

    def _handle_previous_data(self, current_coco, path_previous_coco, images_dir_previous, images_dir_current):

        with open(path_previous_coco, 'r') as f:
            previous_coco = json.load(f)

        # copy all '.jpg' files from previous images dir to current images dir:
        for file in os.listdir(images_dir_previous):
            if file.endswith('.jpg'):
                shutil.copy(os.path.join(images_dir_previous, file), images_dir_current)
                print(f'Copied: {images_dir_current}/{file}')
        print (f'Copied all images from {images_dir_previous} to {images_dir_current}')

        # Update coco with previous data:
        current_coco['annotations'] += previous_coco['annotations']
        current_coco['images'] += previous_coco['images']
        current_coco['videos'] += previous_coco['videos']

        return current_coco

    def move_images(self, image_list, source_dir, dest_dir):
        self.validate_output_path(dest_dir)
        for img in image_list:
            shutil.move(os.path.join(source_dir, img['file_name']),
                        os.path.join(dest_dir, img['file_name']))
            print(f'Moved {img["file_name"]} to {dest_dir}')

    def split_data_and_images(self, split_ratio=0.8, seed = 42, path_previous_coco_train = None, path_previous_coco_test = None, path_previous_images_train = None, path_previous_images_test = None):
        """
        Splits the data into training and testing sets based on the provided split ratio,
        moves the corresponding images to train/test subdirectories, and saves the coco json files.

        :param split_ratio: A float representing the ratio of the split; default is 0.8 for 80% train, 20% test.
        """

        # Set seed, Shuffle images before splitting
        random.seed(seed)
        random.shuffle(self.coco_format['images'])

        # Calculate the split index
        split_index = int(len(self.coco_format['images']) * split_ratio)

        # Split the images into training and testing
        train_images_list = self.coco_format['images'][:split_index]
        test_images_list = self.coco_format['images'][split_index:]

        # Correspondingly split annotations based on the split images
        train_annotations = [ann for ann in self.coco_format['annotations'] if
                             ann['image_id'] in [img['id'] for img in train_images_list]]
        test_annotations = [ann for ann in self.coco_format['annotations'] if
                            ann['image_id'] in [img['id'] for img in test_images_list]]

        # Create two separate coco_format dictionaries for train and test
        train_coco = {
            "videos": self.coco_format['videos'].copy(),
            "images": train_images_list,
            "annotations": train_annotations,
            "categories": self.coco_format['categories']
        }
        test_coco = {
            "videos": self.coco_format['videos'].copy(),
            "images": test_images_list,
            "annotations": test_annotations,
            "categories": self.coco_format['categories']
        }

        # Create train/test subdirectories for images
        train_images_dir = os.path.join(self.output_dir, 'train2017')
        test_images_dir = os.path.join(self.output_dir, 'val2017')
        self.move_images(image_list = train_images_list, source_dir = os.path.join(self.output_dir, 'frames'), dest_dir = os.path.join(self.output_dir, 'train2017'))
        self.move_images(image_list = test_images_list, source_dir = os.path.join(self.output_dir, 'frames'), dest_dir = os.path.join(self.output_dir, 'val2017'))

        # Add the annotations and images from the previous train/test data:
        if path_previous_coco_train and path_previous_coco_test and path_previous_images_train and path_previous_images_test:

            train_coco = self._handle_previous_data(current_coco = train_coco,
                                              path_previous_coco = path_previous_coco_train,
                                              images_dir_previous = path_previous_images_train,
                                              images_dir_current = train_images_dir)

            test_coco = self._handle_previous_data(current_coco = test_coco,
                                              path_previous_coco = path_previous_coco_test,
                                              images_dir_previous = path_previous_images_test,
                                              images_dir_current = test_images_dir)

        # Save the coco_format JSON files
        today_date = datetime.now().strftime("%d%m%Y")
        train_coco_path = os.path.join(self.output_dir, 'annotations', f'coco_train_{today_date}.json')
        test_coco_path = os.path.join(self.output_dir, 'annotations', f'coco_test_{today_date}.json')

        self.save_coco_json(data=train_coco, file_path = train_coco_path)
        self.save_coco_json(data=test_coco, file_path = test_coco_path)

        return train_coco_path, test_coco_path


if __name__ == '__main__':

    # Download files from S3:
    S3_PATHS_LIST = ['s3://fruitspec.dataset/object-detection/JAI/ISRAEL/MANDAR/MEIRAVVA/091123/',
                     's3://fruitspec.dataset/object-detection/JAI/ISRAEL/ORANGE/SUMMERG0/091123/',
                     's3://fruitspec.dataset/object-detection/JAI/ISRAEL/ORANGE/SUMMERG0/121123/',
                     's3://fruitspec.dataset/object-detection/JAI/ISRAEL/ORANGE/SUMMERG0/151123/',
                     's3://fruitspec.dataset/object-detection/JAI/ISRAEL/ORANGE/RAUSTENB/161123/',
                     's3://fruitspec.dataset/object-detection/JAI/ISRAEL/ORANGE/DEMOLTMX/161123/',
                     's3://fruitspec.dataset/object-detection/JAI/ISRAEL/MANDAR/MEIRAVVA/151123/',
                     's3://fruitspec.dataset/object-detection/JAI/SAXXXX/APPLEX/']


    #OUTPUT_DATA_DIR = '/home/fruitspec-lab-3/FruitSpec/Data/Counter/CLAHE_FSI'
    OUTPUT_DATA_DIR = '/home/lihi/FruitSpec/Data/CLAHE_FSI'
    LIST_OF_FILES_TO_DOWNLOAD = ['tracks.csv', 'Result_FSI.mkv', 'FSI_CLAHE.mkv']
    OUTPUT_RESULTS_DIR = os.path.join(OUTPUT_DATA_DIR, 'debug_train_test_mandar')
    ROTATE = 'counter_clockwise'

    for S3_PATH in S3_PATHS_LIST:

        # Download files from S3:
        block_name = get_subpath_from_dir(S3_PATH, dir_name ="JAI", include_dir=False)
        ROWS_FOLDER_LOCAL = os.path.join(OUTPUT_DATA_DIR, block_name)
        #download_s3_files(S3_PATH, ROWS_FOLDER_LOCAL, string_param= LIST_OF_FILES_TO_DOWNLOAD, skip_existing=True, save_flat=False)

    ###############################################################################################################################################
    # Get a list of all rows dir paths (where there are tracks.csv files):
    # Its is done like that (and not directly on all downloaded files from s3) because that we need to manually remove unwanted rows

    OUTPUT_DATA_DIR = '/home/lihi/FruitSpec/Data/CLAHE_FSI/ISRAEL/MANDAR'

    local_rows_dirs = find_subdirs_with_file(OUTPUT_DATA_DIR, file_name = 'tracks.csv', return_dirs=True, single_file=False)
    local_rows_dirs = list(set([x.rsplit('/', 2)[0] for x in local_rows_dirs])) # get rows paths

    pipeline = TaggingPipeline(output_dir=OUTPUT_RESULTS_DIR,rotate_option=ROTATE, frames_interval=10)

    for ROWS_FOLDER_LOCAL in tqdm(local_rows_dirs):
        pipeline.run(videos_folder = ROWS_FOLDER_LOCAL, save_frames=False, update_coco=True, video_name = 'Result_FSI.mkv') # Save coco from old FSI
        pipeline.run(videos_folder = ROWS_FOLDER_LOCAL, save_frames=True, update_coco=False, video_name = 'FSI_CLAHE.mkv')  # Save frames from new FSI (CLAHE)

    # split data to train and test, and update existing dataset:
    train_images_dir_prev = r'/home/lihi/FruitSpec/Data/CLAHE_FSI/debug_train_test_orange/train2017'
    test_images_dir_prev = r'/home/lihi/FruitSpec/Data/CLAHE_FSI/debug_train_test_orange/val2017'
    train_coco_prev = r'/home/lihi/FruitSpec/Data/CLAHE_FSI/debug_train_test_orange/annotations/coco_train_20231123.json'
    test_coco_prev = r'/home/lihi/FruitSpec/Data/CLAHE_FSI/debug_train_test_orange/annotations/coco_test_20231123.json'

    pipeline.split_data_and_images(split_ratio=0.85, seed = 42, path_previous_coco_train = train_coco_prev, path_previous_coco_test = test_coco_prev, path_previous_images_train = train_images_dir_prev, path_previous_images_test = test_images_dir_prev)

    print('Done')




