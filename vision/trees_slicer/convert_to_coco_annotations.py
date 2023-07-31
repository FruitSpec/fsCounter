import cv2
import os
import json
import pandas as pd
import json
from collections import Counter
from vision.trees_slicer.split_data_coco_format_to_tain_val import split_train_val_images
from vision.tools.utils_general import find_subdirs_with_file
from vision.tools.manual_slicer import slice_to_trees

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

    def update_annotation_coco_dict(self, image_id, bbox, id_bbox):
        x, y, w, h = bbox
        annotation_dict = {
            "id": id_bbox,
            "image_id": image_id,
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
                                should_save_frames=True, save_annotated_video = False):
    """
    This script display video with annotations,
    saves frames from a video and creates a coco annotations file for the frames.
    If the video already exists in the annotations file, it will not be added again.
    """

    camera = 'zed' if 'zed' in INPUT_VIDEO_PATH.lower() else 'jai'
    frame_width = 1080 if camera =='zed' else 1536
    frame_height = 1920 if camera =='zed' else 2048

    # get bbox data:
    annotations_file_name = os.path.basename(ANNOTATIONS_FILE_PATH)
    if annotations_file_name == 'all_slices.csv':
        df = pd.read_csv(ANNOTATIONS_FILE_PATH, index_col = 0)
    else:
        df, _ = slice_to_trees(ANNOTATIONS_FILE_PATH, video_path=INPUT_VIDEO_PATH, resize_factor=3, output_path=None,
                                 h=frame_height, w=frame_width, on_fly=True)

    df = add_bbox_to_slice_trees(df, frame_width = frame_width, frame_height = frame_height)


    # check if video exist:
    if not os.path.exists(INPUT_VIDEO_PATH):
        raise Exception(f"Video {INPUT_VIDEO_PATH} does not exist")

    video_name = '_'.join(INPUT_VIDEO_PATH.split('.')[0].split('/')[-5:])


    # video
    cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
    fps = cap.get(cv2.CAP_PROP_FPS)

    if save_annotated_video == True:
        fourcc = cv2.VideoWriter_fourcc(*'X264')
        output_video_path = INPUT_VIDEO_PATH.split('.')[0] + '_annotated.mkv'
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))


    paused = False  # Flag to determine if video display is paused

    # if annotations file exists, load it
    coco = CocoAnnotationsUpdater(COCO_ANNOTATIONS_PATH)

    if len(coco.coco_dict['images']) == 0:
        image_id = 0
        bbox_id = 0
    else: # get the maximal image id exist in the json
        image_id = max(image['id'] for image in coco.coco_dict['images']) + 1
        bbox_id = max(annotation['id'] for annotation in coco.coco_dict['annotations'])

    output_img_name_previous = None

    if not coco.video_exist_in_json(video_name):
        while cap.isOpened():
            if not paused:
                ret, frame = cap.read()

                if camera == "jai":
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    frame = cv2.cvtColor(frame , cv2.COLOR_RGB2BGR)

                if ret:
                    frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1  # Get the next frame ID

                    if frame_id % 30 == 0:  # skip frames (frames 29 == 30 because of a bug)
                        frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

                    output_img_name = f"{video_name}_f{frame_id}.jpg"
                    #############################3
                    print(output_img_name)
                    if output_img_name == output_img_name_previous:
                        print (f"frame {frame_id} already exist in the json")
                    output_img_name_previous = output_img_name
                    ############################
                    # Filter dataframe for the current frame ID
                    frame_data = df[df['frame_id'] == frame_id]

                    if not frame_data.empty:
                        if should_save_frames:  # Save frames that has annotations
                            save_frames(frame, OUTPUT_FRAMES_PATH, output_img_name)

                        coco.update_image_coco_dict(output_img_name, image_id, frame)

                        for _, row in frame_data.iterrows():
                            x, y, w, h = int(row['x']), int(row['y']), int(row['w']), int(row['h'])

                            # update annotation in coco dict
                            coco.update_annotation_coco_dict(image_id, bbox=[x, y, w, h],
                                                                         id_bbox=bbox_id)
                            bbox_id += 1

                            # Draw bounding boxes on the frame
                            if row['tree_id'] % 2 == 0:
                                bbox_color = (0, 255, 0)
                            else:
                                bbox_color = (255, 0, 255)
                            cv2.rectangle(frame, (x, y), (x + w, y + h), bbox_color, 15)  # Draw the bounding box
                            # write row['tree_id'] on the bbox:
                            cv2.putText(frame, f"Tree_id: {str(row['tree_id'])}", (x+30, y+30 ), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                                        color=(255, 0, 255), thickness=2)


                        image_id += 1

                    # Write frame number on the frame
                    cv2.putText(frame, f'Frame: {frame_id}', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                                color=(255, 0, 255), thickness=2)
                    resized_frame = cv2.resize(frame, None, fx=1 / 2, fy=1 / 2)
                    cv2.imshow('Frame', resized_frame)

                    if save_annotated_video == True:
                        out.write(frame)

                # Pause/resume video display on space key press
                key = cv2.waitKey(80)
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
        if save_annotated_video:
            out.release()
            print(f'Annotated video saved to: {output_video_path}')
        cap.release()
        cv2.destroyAllWindows()

def count_images_in_coco_file(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)

    image_names = [image['file_name'] for image in data['images']]
    image_counts = Counter(image_names)

    num_images = len(data['images'])
    duplicate_images = [name for name, count in image_counts.items() if count > 1]
    num_duplicates = len(duplicate_images)

    print(f"Total number of images: {num_images}")
    print(f"Number of duplicate images: {num_duplicates}")
    print("Duplicate image names:", duplicate_images)

    return num_images, num_duplicates, duplicate_images

def save_frames_and_annotations_scraping_dirs(dir_path, output_frames_path, COCO_ANNOTATIONS_PATH, file_name = 'trees_manual_annotations_'):
    '''
    itterate over folders, and preform 'save_frames_and_annotations' for the folders containing the desired file_name
    (file_name = 'trees_manual_annotations_')
    '''
    annotation_files = find_subdirs_with_file(dir_path, file_name = file_name, return_dirs=False, single_file=False)
    for file_path in annotation_files:
        print(file_path)
        video_path = os.path.join(os.path.dirname(file_path), 'zed_rgd.avi')
        save_frames_and_annotations(file_path, video_path, output_frames_path, COCO_ANNOTATIONS_PATH,should_save_frames=True)
    print('Finished generating dataset')


if __name__ == '__main__':

    ANNOTATIONS_FILE_PATH = "/home/lihi/FruitSpec/code/lihi/fsCounter/vision/trees_slicer/slice_by_distance_using_tx_translations/data/roi_row_debug/all_slices.csv"
    INPUT_VIDEO_PATH = "/home/lihi/FruitSpec/code/lihi/fsCounter/vision/trees_slicer/slice_by_distance_using_tx_translations/data/roi_row_debug/Result_RGB.mkv"
    OUTPUT_FRAMES_PATH = "/home/lihi/FruitSpec/code/lihi/fsCounter/vision/trees_slicer/slice_by_distance_using_tx_translations/data/roi_row_debug/all_images"
    COCO_ANNOTATIONS_PATH = '/home/lihi/FruitSpec/code/lihi/fsCounter/vision/trees_slicer/slice_by_distance_using_tx_translations/data/roi_row_debug/all_annotations.json'

    save_frames_and_annotations(ANNOTATIONS_FILE_PATH, INPUT_VIDEO_PATH, OUTPUT_FRAMES_PATH, COCO_ANNOTATIONS_PATH, should_save_frames=False, save_annotated_video= True)
    print ('done')

#############################################################################
    # # Itterate subfolders, save frames and coco annotations:
    # FOLDER_PATH = '/home/fruitspec-lab-3/FruitSpec/Data/customers/DEWAGD'
    # OUTPUT_FRAMES_PATH = "/home/fruitspec-lab-3/FruitSpec/Data/customers/DEWAGD/training_yoloX/slicer_data_rgd/all_images"
    # COCO_ANNOTATIONS_PATH = '/home/fruitspec-lab-3/FruitSpec/Data/customers/DEWAGD/training_yoloX/slicer_data_rgd/annotations/all_annotations.json'
    #
    # save_frames_and_annotations_scraping_dirs(FOLDER_PATH, OUTPUT_FRAMES_PATH, COCO_ANNOTATIONS_PATH, file_name = 'trees_manual_annotations_')
    #
    # num_images, num_duplicates, duplicate_images = count_images_in_coco_file(COCO_ANNOTATIONS_PATH)
    #
    # split_train_val_images(OUTPUT_FRAMES_PATH, COCO_ANNOTATIONS_PATH, train_ratio=0.8)
    #
    # print ('Done')
    ###########################################################################################

    # ANNOTATIONS_FILE_PATH = "/home/lihi/FruitSpec/Data/customers/DEWAGD/190123/DWDBLE43/R10B/trees_manual_annotations_R10B.json"
    # INPUT_VIDEO_PATH = "/home/lihi/FruitSpec/Data/customers/DEWAGD/190123/DWDBLE43/R10B/zed_rgd.avi"
    # OUTPUT_FRAMES_PATH = "/home/lihi/FruitSpec/Data/training_yoloX/slicer_data_rgd/all_images"
    # COCO_ANNOTATIONS_PATH = '/home/lihi/FruitSpec/Data/training_yoloX/slicer_data_rgd/annotations/all_annotations.json'

    # ANNOTATIONS_FILE_PATH = "/home/fruitspec-lab-3/FruitSpec/Data/customers/DEWAGD/190123/DWDBLE33/R21B/trees_manual_annotations_R21B.json"
    # INPUT_VIDEO_PATH = "/home/fruitspec-lab-3/FruitSpec/Data/customers/DEWAGD/190123/DWDBLE33/R21B/zed_rgd.avi"
    # OUTPUT_FRAMES_PATH = "/home/fruitspec-lab-3/FruitSpec/Data/customers/DEWAGD/training_yoloX/slicer_data_rgd/all_images"
    # COCO_ANNOTATIONS_PATH = '/home/fruitspec-lab-3/FruitSpec/Data/customers/DEWAGD/training_yoloX/slicer_data_rgd/annotations/all_annotations.json'

    # save_frames_and_annotations(ANNOTATIONS_FILE_PATH,INPUT_VIDEO_PATH, OUTPUT_FRAMES_PATH, COCO_ANNOTATIONS_PATH)
    # print ('done')

