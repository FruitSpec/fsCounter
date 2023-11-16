import cv2
import pandas as pd
import json
import os


class TaggingPipeline:
    def __init__(self, video_path, tracks_csv_path, output_dir, video_identifier, rotate_option=None,
                 frames_interval=10):
        self.video_path = video_path
        self.tracks_csv_path = tracks_csv_path
        self.output_dir = output_dir
        self.video_identifier = video_identifier
        self.rotate_option = rotate_option
        self.frames_interval = frames_interval

    @staticmethod
    def validate_output_path(path):
        if not os.path.exists(path):
            os.makedirs(path)

    def frames_idx(self, frames_amount):
        return [x for x in range(frames_amount) if x % self.frames_interval == 0]

    def load_video(self):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print("Error opening video stream or file")
            return None
        return cap

    def rotate_frame(self, frame):
        if self.rotate_option == 'clockwise':
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif self.rotate_option == 'counter_clockwise':
            frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    def save_frames(self, cap, frames_idx_list):
        frame_save_path = os.path.join(self.output_dir, 'frames')
        self.validate_output_path(frame_save_path)

        for frame_id in frames_idx_list:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
            if ret:
                frame = self.rotate_frame(frame)
                frame_file_path = os.path.join(frame_save_path, f"{self.video_identifier}_frame_{frame_id}.jpg")
                cv2.imwrite(frame_file_path, frame)
                print(f'Saved: {frame_file_path}')
            else:
                print(f"Failed to capture frame {frame_id}")

    def filter_tracks_by_frame_ids(self, frames_idx_list):
        tracking_results = pd.read_csv(self.tracks_csv_path)
        return tracking_results[tracking_results['frame_id'].isin(frames_idx_list)]

    def tracks_to_coco_json(self, df):
        coco_format = {
            "images": [],
            "annotations": [],
            "categories": [{
                "id": 1,
                "name": "object",
                "supercategory": "none"
            }]
        }

        image_ids = set()

        for _, row in df.iterrows():
            image_id = int(row['frame_id'])
            if image_id not in image_ids:
                image = {
                    "id": image_id,
                    "width": None,  # Width is not available in the provided data
                    "height": None,  # Height is not available in the provided data
                    "file_name": f"{self.video_identifier}_frame_{image_id}.jpg"
                }
                coco_format["images"].append(image)
                image_ids.add(image_id)

            annotation = {
                "id": len(coco_format["annotations"]) + 1,
                "image_id": image_id,
                "category_id": 1,
                "bbox": [row['x1'], row['y1'], row['x2'] - row['x1'], row['y2'] - row['y1']],
                "area": (row['x2'] - row['x1']) * (row['y2'] - row['y1']),
                "iscrowd": 0,
                "segmentation": [],  # Segmentation is not available
                "score": row['obj_conf']  # Assuming obj_conf is the confidence score
            }
            coco_format["annotations"].append(annotation)

        return json.dumps(coco_format, indent=2)

    def save_coco_json(self, coco_json):
        self.validate_output_path(self.output_dir)
        output_path_coco = os.path.join(self.output_dir, f"coco_{self.video_identifier}.json")
        with open(output_path_coco, 'w') as file:
            file.write(coco_json)
        print(f'Saved: {output_path_coco}')

    def run(self, save_frames=True, save_coco=True):
        cap = self.load_video()
        if cap is None:
            return

        tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frames_idx_list = self.frames_idx(tot_frames)

        if save_frames:
            self.save_frames(cap, frames_idx_list)

        if save_coco:
            filtered_tracks_df = self.filter_tracks_by_frame_ids(frames_idx_list)
            coco_json = self.tracks_to_coco_json(filtered_tracks_df)
            self.save_coco_json(coco_json)

        cap.release()
        print('Pipeline execution completed')


if __name__ == '__main__':


    VIDEO_PATH = '/home/lihi/FruitSpec/Data/CLAHE_FSI/MANDAR/MEIRAVVA/091123/row_1/1/Result_FSI.mkv'
    TRACKS_CSV_PATH ='/home/lihi/FruitSpec/Data/CLAHE_FSI/MANDAR/MEIRAVVA/091123/row_1/1/tracks.csv'
    OUTPUT_DIR = '/home/lihi/FruitSpec/Data/CLAHE_FSI/DeleteMe/'
    VIDEO_IDENTIFIER = 'video_identifier'
    ROTATE = 'counter_clockwise'


    pipeline = TaggingPipeline(
        video_path=VIDEO_PATH,
        tracks_csv_path=TRACKS_CSV_PATH,
        output_dir=OUTPUT_DIR,
        video_identifier=VIDEO_IDENTIFIER,
        rotate_option=ROTATE)

    pipeline.run(save_frames=True, save_coco=True)


