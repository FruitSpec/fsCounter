import cv2
import pandas as pd
import os
from vision.visualization.drawer import draw_rectangle, draw_text, draw_highlighted_test, get_color
import numpy as np
from vision.misc.help_func import get_repo_dir, load_json, validate_output_path



def draw_dets(frame, dets, t_index=6):
    """
    Draws bounding boxes and track IDs on the input frame for each detection.

    Args:
        frame (numpy.ndarray): The input frame to draw on.
        dets (numpy.ndarray): A 2D array of detections, where each row is a detection and the columns contains the
            coordinates of the bounding box and the track ID.
        t_index (int, optional): The index of the track ID column in the detections array. Defaults to 6.

    Returns:
        numpy.ndarray: The input frame with bounding boxes and track IDs drawn on it.
    """
    for det in dets:
        track_id = det[t_index]
        color_id = int(track_id) % 15  # 15 is the number of colors in list
        color = get_color(color_id)
        text_color = get_color(-1)
        frame = draw_rectangle(frame, (int(det[0]), int(det[1])), (int(det[2]), int(det[3])), color, 3)
        frame = draw_highlighted_test(frame, f'ID:{int(track_id)}', (det[0], det[1]), frame.shape[1], color, text_color,
                                      True, 10, 3)
    return frame


def write_vid(fsi, tracks_df, s_frame=0, max_frame=np.inf, t_index=6, frame_save_directory=""):
    """
    Writes a new video file with the input tracks_df overlaid on top of the input video file.

    Args:
        fsi (cv2.VideoCapture): The input video file to overlay tracks on top of.
        tracks_df (pandas.DataFrame): A DataFrame containing track data with columns "frame", "x1", "y1", "x2",
            "y2", and "track_id".
        s_frame (int, optional): The starting frame of the output video. Defaults to 0.
        max_frame (int, optional): The maximum number of frames to write to the output video. Defaults to np.inf.
        t_index (int, optional): The index of the track ID column in the detections array. Defaults to 6.
        frame_save_directory (str, optional): path for saving frame by frame images
    """
    i = s_frame
    n = min(cap_fsi.get(cv2.CAP_PROP_FRAME_COUNT), max_frame)
    while cap_fsi.isOpened() and i < max_frame:
        print(f"\r{i}/{n - 1} ({i / (n - s_frame) * 100: .2f}%) frames", end="")
        _, frame_fsi = cap_fsi.read()
        if isinstance(frame_fsi, type(None)):
            break
        if rotate:
            frame_fsi = cv2.rotate(frame_fsi, cv2.ROTATE_90_COUNTERCLOCKWISE)
        dets = tracks_df[tracks_df["frame"] == i].to_numpy()
        if len(dets):
            frame_fsi = draw_dets(frame_fsi, dets, t_index=t_index)
        if frame_save_directory != "":
            cv2.imwrite(os.path.join(frame_save_directory, f"frame_{i}.png"), frame_fsi)
        fsi.write(frame_fsi)
        i += 1
    fsi.release()


def make_vid_with_dets(cap_fsi, tracks_df, rotate, fps=0, new_vid_name="", s_frame=0, max_frame=np.inf
                       , frame_save_directory=""):
    """
    Creates a new video file with detected object tracks overlaid on each frame.

    Args:
        cap_fsi (cv2.VideoCapture): The input video capture object.
        tracks_df (pd.DataFrame): A DataFrame containing object tracks with columns "frame", "x1", "y1", "x2", "y2",
         and "track_id".
        rotate (bool): If True, rotate the video frames by 90 degrees counterclockwise before processing.
        fps (int): The frames per second to use for the output video. If 0, use the same frame rate as the input video.
        new_vid_name (str): The filename to use for the output video. If empty, create a default filename.
        s_frame (int): The index of the starting frame.
        max_frame (int): The maximum number of frames to process.
        frame_save_directory (str, optional): path for saving frame by frame images

    Returns:
        None
    """
    width = int(cap_fsi.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_fsi.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if rotate:
        width, height = height, width
    if not fps:
        fps = int(cap_fsi.get(cv2.CAP_PROP_FPS))
    if new_vid_name == "":
        new_vid_name = os.path.join(row_path, f'Result_FSI_with_dets.mkv')
    else:
        new_vid_name = os.path.join(row_path, new_vid_name)
    if frame_save_directory != "":
        validate_output_path(frame_save_directory)
    fsi = cv2.VideoWriter(new_vid_name, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (width, height))
    cap_fsi.set(cv2.CAP_PROP_POS_FRAMES, s_frame)
    write_vid(fsi, tracks_df, s_frame, max_frame, frame_save_directory=frame_save_directory)


if __name__ == "__main__":
    row_path = r"/media/fruitspec-lab/cam175/customers_new/PROPAL/Mandarin/250323/row_7/1"
    rotate = True
    s_frame = 0
    max_frame = 30
    fps = 15
    new_vid_name = ""
    frame_save_directory = ""

    jai_fp = os.path.join(row_path, f'Result_FSI.mkv')
    tracks_path = os.path.join(row_path, f'tracks.csv')
    tracks_df = pd.read_csv(tracks_path)
    cap_fsi = cv2.VideoCapture(jai_fp)

    make_vid_with_dets(cap_fsi, tracks_df, rotate, fps=fps, new_vid_name=new_vid_name, s_frame=s_frame,
                       max_frame=max_frame, frame_save_directory=frame_save_directory)



