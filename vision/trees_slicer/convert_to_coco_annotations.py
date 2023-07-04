import cv2
from vision.tools.manual_slicer import slice_to_trees
import os
import json

def add_bbox_to_slice_trees(df, frame_width = 1080, frame_height = 1920):
    # Add bbox coco format [top left x position, top left y position, width, height]
    df.loc[df['start'] == -1, 'start'] = 0
    df['x'] = df['start']
    df['y'] = 0
    df.loc[df['end'] == -1, 'end'] = frame_width
    df['w'] = df['end'] - df['start']
    df['h'] = frame_height

    # Remove un-correct rows artifact of the last tree
    highest_tree_id = df['tree_id'].max()
    df = df[~((df['tree_id'] == highest_tree_id) & (df['x'] == 0) & (df['y'] == 0))]
    return df


def save_frames(frame, output_dir, output_img_name):  # Save frames that has annotations
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_img_name)
    cv2.imwrite(output_path, frame)
    print(f"saved frame: {output_path}")

class CocoAnnotationsUpdater:
    def __init__(self, COCO_ANNOTATIONS_PATH):
        self.COCO_ANNOTATIONS_PATH = COCO_ANNOTATIONS_PATH
        self.coco_dict = self.load_coco_annotations()

    def load_coco_annotations(self):
        '''
        Load coco annotations file, if not exist create new one
        '''
        if os.path.exists(self.COCO_ANNOTATIONS_PATH):
            with open(self.COCO_ANNOTATIONS_PATH, 'r') as f:
                coco_dict = json.load(f)
        else:
            coco_dict = {
                "images": [],
                "annotations": [],
                "categories": [{"id": 0, "name": "tree", "supercategory": "none"}],
                "info": { "videos": []}}
        return coco_dict

    def update_image_coco_dict(self, output_img_name, image_id, frame):
        img_dict = {
            "id": image_id,
            "license": 1,
            "file_name": output_img_name,
            "height": frame.shape[0],
            "width": frame.shape[1],
            "date_captured": "" }
        self.coco_dict["images"].append(img_dict)
        return self.coco_dict

    def update_video_coco_dict(self, video_name):
        if "videos" not in self.coco_dict["info"]:
            self.coco_dict["info"]["videos"] = []
        self.coco_dict['info']['videos'].append(video_name)
        return self.coco_dict

    def update_annotation_coco_dict(self, output_img_name, bbox, id_bbox):
        x, y, w, h = bbox
        annotation_dict = {
            "id": id_bbox,
            "image_id": output_img_name,
            "category_id": 0,
            "bbox": [x, y, w, h],
            "area": w * h,
            "segmentation": [],
            "iscrowd": 0}

        self.coco_dict["annotations"].append(annotation_dict)
        return self.coco_dict

    def video_exist_in_json(self, video_name):
        if video_name in self.coco_dict['info']['videos']:
            print(f"Video {video_name} already exists in the annotations file")
            return True
        else:
            return False

    def save_annotations_json(self):
        os.makedirs(os.path.dirname(self.COCO_ANNOTATIONS_PATH), exist_ok=True)
        with open(self.COCO_ANNOTATIONS_PATH, 'w') as f:
            json.dump(self.coco_dict, f)
            print(f'Saved annotations to {COCO_ANNOTATIONS_PATH}')

def save_frames_and_annotations(ANNOTATIONS_FILE_PATH, INPUT_VIDEO_PATH, OUTPUT_FRAMES_PATH, COCO_ANNOTATIONS_PATH,
                                should_save_frames=True):
    video_name = '_'.join(INPUT_VIDEO_PATH.split('.')[0].split('/')[-5:])

    df, df2 = slice_to_trees(ANNOTATIONS_FILE_PATH, video_path=INPUT_VIDEO_PATH, resize_factor=3, output_path=None,
                             h=1920, w=1080, on_fly=True)
    df = add_bbox_to_slice_trees(df)

    # video
    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    paused = False  # Flag to determine if video display is paused

    # if annotations file exists, load it
    coco = CocoAnnotationsUpdater(COCO_ANNOTATIONS_PATH)

    if len(coco.coco_dict['images']) == 0:
        image_id = 0
        bbox_id = 0
    else: # get the maximal image id exist in the json
        image_id = max(image['id'] for image in coco.coco_dict['images']) + 1
        bbox_id = max(annotation['id'] for annotation in coco.coco_dict['annotations'])

    if not coco.video_exist_in_json(video_name):
        while cap.isOpened():
            if not paused:
                ret, frame = cap.read()

                if ret:
                    frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1  # Get the current frame ID

                    if frame_id % 30 == 0:  # skip frames (frames 29 == 30 because of a bug)
                        frame_id += 1

                    output_img_name = f"{video_name}_f{frame_id}.jpg"

                    # Filter dataframe for the current frame ID
                    frame_data = df[df['frame_id'] == frame_id]

                    if not frame_data.empty:
                        if should_save_frames:  # Save frames that has annotations
                            save_frames(frame, OUTPUT_FRAMES_PATH, output_img_name)

                        coco.update_image_coco_dict(output_img_name, image_id, frame)
                        image_id += 1

                        for _, row in frame_data.iterrows():
                            x, y, w, h = int(row['x']), int(row['y']), int(row['w']), int(row['h'])

                            # update annotation in coco dict
                            coco.update_annotation_coco_dict(output_img_name, bbox=[x, y, w, h],
                                                                         id_bbox=bbox_id)
                            bbox_id += 1

                            # Draw bounding boxes on the frame
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 15)  # Draw the bounding box

                    # Write frame number on the frame
                    cv2.putText(frame, f'Frame: {frame_id}', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                                color=(255, 0, 255), thickness=2)
                    resized_frame = cv2.resize(frame, None, fx=1 / 2, fy=1 / 2)
                    cv2.imshow('Frame', resized_frame)

                # Pause/resume video display on space key press
                key = cv2.waitKey(30)
                if key == ord(' '):
                    paused = not paused

                elif key & 0xFF == ord('q'):
                    break

                if df['frame_id'].iloc[-1] == frame_id:  # if last annotation:
                    coco.update_video_coco_dict(video_name)
                    coco.save_annotations_json()
                    break

            else:
                # Continue to check for space key press to resume video display
                key = cv2.waitKey(1)
                if key == ord(' '):
                    paused = not paused

        # Release the video capture object and close the windows
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    """
    This script display video with annotations, 
    saves frames from a video and creates a coco annotations file for the frames.
    If the video already exists in the annotations file, it will not be added again.
    """
    ANNOTATIONS_FILE_PATH = "/home/lihi/FruitSpec/Data/customers/DEWAGD/190123/DWDBLE43/R10B/trees_manual_annotations_R10B.json"
    INPUT_VIDEO_PATH = "/home/lihi/FruitSpec/Data/customers/DEWAGD/190123/DWDBLE43/R10B/zed_rgd.avi"
    OUTPUT_FRAMES_PATH = "/home/lihi/FruitSpec/Data/training_yoloX/slicer_data_rgd/all_images"
    COCO_ANNOTATIONS_PATH = '/home/lihi/FruitSpec/Data/training_yoloX/slicer_data_rgd/annotations/all_annotations.json'

    # ANNOTATIONS_FILE_PATH = "/home/fruitspec-lab-3/FruitSpec/Data/customers/DEWAGD/190123/DWDBLE33/R21B/trees_manual_annotations_R21B.json"
    # INPUT_VIDEO_PATH = "/home/fruitspec-lab-3/FruitSpec/Data/customers/DEWAGD/190123/DWDBLE33/R21B/zed_rgd.avi"
    # OUTPUT_FRAMES_PATH = "/home/fruitspec-lab-3/FruitSpec/Data/customers/DEWAGD/training_yoloX/slicer_data_rgd/all_images"
    # COCO_ANNOTATIONS_PATH = '/home/fruitspec-lab-3/FruitSpec/Data/customers/DEWAGD/training_yoloX/slicer_data_rgd/annotations/all_annotations.json'

    save_frames_and_annotations(ANNOTATIONS_FILE_PATH,INPUT_VIDEO_PATH, OUTPUT_FRAMES_PATH, COCO_ANNOTATIONS_PATH)
    print ('done')

