import cv2
import pandas as pd
import json
import os
from tqdm import tqdm
from vision.tools.utils_general import find_subdirs_with_file

class TaggingPipeline:
    """
    A utility for processing videos and tracking data to extract frames and compile annotations into a COCO-format JSON.

    Usage:
        - Initialize with paths to video folders and output directory.
        - Run the pipeline specifying whether to save frames and/or update COCO JSON.
        - Outputs frames and a COCO JSON with annotations for object detection models.

    The video_identifier is created from the last five subdirectories of the video's path.
    """

    def __init__(self, videos_folder, output_dir, rotate_option=None, frames_interval=10):
        self.videos_folder = videos_folder
        self.output_dir = output_dir
        self.rotate_option = rotate_option
        self.frames_interval = frames_interval
        self.coco_format = {
            "images": [],
            "annotations": [],
            "categories": [{
                "id": 1,
                "name": "object",
                "supercategory": "none"
            }]
        }
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
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video stream or file: {video_path}")
            return

        tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_idx_list = self.frames_idx(tot_frames, self.frames_interval)
        frame_save_path = os.path.join(self.output_dir, 'frames')
        self.validate_output_path(frame_save_path)

        # Save frames with a progress bar
        if save_frames:
            for frame_id in tqdm(frames_idx_list, unit="frame"):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                ret, frame = cap.read()
                if ret:
                    frame = self.rotate_frame(frame)
                    frame_file_path = os.path.join(frame_save_path, f"{video_identifier}_frame_{frame_id}.jpg")
                    cv2.imwrite(frame_file_path, frame)
                else:
                    print(f"Failed to capture frame {frame_id}")

        cap.release()

        # Process tracks and update COCO JSON with a progress bar
        if update_coco:
            tracking_results = pd.read_csv(tracks_csv_path)
            filtered_tracks = tracking_results[tracking_results['frame_id'].isin(frames_idx_list)]
            self.update_coco_json(filtered_tracks, video_identifier)


    def update_coco_json(self, df, video_identifier):
        for _, row in df.iterrows():
            image_id = int(row['frame_id'])
            image_file_name = f"{video_identifier}_frame_{image_id}.jpg"

            # Only add the image if it's not already in the coco_format
            if not any(image['file_name'] == image_file_name for image in self.coco_format['images']):
                image = {
                    "id": image_id,
                    "width": None,  # Width is not available in the provided data
                    "height": None,  # Height is not available in the provided data
                    "file_name": image_file_name
                }
                self.coco_format["images"].append(image)

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

    def save_coco_json(self):
        output_path_coco = os.path.join(self.output_dir, 'coco_dataset.json')
        with open(output_path_coco, 'w') as file:
            json.dump(self.coco_format, file, indent=2)
        print(f'Saved: {output_path_coco}')

    def run(self, save_frames=True, update_coco=True, video_name = 'Result_FSI.mkv'):

        subdirs = find_subdirs_with_file(self.videos_folder, file_name = video_name, return_dirs=True, single_file=False)
        for subdir in subdirs:

            video_path = os.path.join(subdir, video_name)
            tracks_csv_path = os.path.join(subdir, 'tracks.csv')
            video_identifier = '_'.join(subdir.split(os.sep)[-5:])

            self.process_video(video_path, tracks_csv_path, video_identifier, save_frames, update_coco)

        if update_coco:
            self.save_coco_json()


if __name__ == '__main__':

    VIDEOS_FOLDER = '/home/lihi/FruitSpec/Data/CLAHE_FSI/MANDAR/MEIRAVVA/091123'
    OUTPUT_DIR = '/home/lihi/FruitSpec/Data/CLAHE_FSI/DeleteMe/'
    ROTATE = 'counter_clockwise'


    pipeline = TaggingPipeline(
        videos_folder = VIDEOS_FOLDER,
        output_dir= OUTPUT_DIR,
        rotate_option=ROTATE)

    pipeline.run(save_frames=False, update_coco=True, video_name = 'Result_FSI.mkv')
    pipeline.run(save_frames=True, update_coco=False, video_name = 'FSI_CLAHE.mkv')

    print('Done')




