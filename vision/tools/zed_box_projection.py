from vision.tools.scripts.mkv_to_frames import folder_to_frames
import os
from vision.tools.sensors_alignment import align_folder
import pandas as pd
import numpy as np
from vision.tools.image_stitching import plot_2_imgs
import cv2
from vision.visualization.drawer import draw_rectangle, draw_text, draw_highlighted_test, get_color
from tqdm import tqdm
def draw_dets(frame, track_id, det, scale = 15):
    color_id = int(track_id) % 15  # 15 is the number of colors in list
    color = get_color(color_id)
    text_color = get_color(-1)
    frame = draw_rectangle(frame, (int(det[0]), int(det[1])), (int(det[2]), int(det[3])), color, 3)
    frame = draw_highlighted_test(frame, f'ID:{track_id}', (det[0], det[1]), frame.shape[1], color, text_color,
                                  True, scale, 5)

    return frame


def frames_to_video(input_folder, output_file, fps=15.0, frame_size=(1536, 2048)):
    out = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), fps, frame_size)
    frame_names = [f for f in os.listdir(input_folder) if f.endswith(".jpg")]
    frame_names.sort(key=lambda x: int(x.split(".")[0].split("_")[-1]))
    frames = [cv2.imread(os.path.join(input_folder, f)) for f in frame_names]
    for frame_number, frame in enumerate(frames):
        print(f"frame: {frame_number}")
        out.write(frame.astype(np.uint8))

    # Release the VideoWriter object
    out.release()
    print("release")

if __name__ == "__main__":
    frame_number = 120
    overwrite = False
    process = True
    folder_path = "/media/yotam/easystore/preTiltDown40cm/relevent_movies"
    meas_path = "/media/yotam/easystore/preTiltDown40cm/measures_zed_3.csv"
    w, h = 1536, 2048
    frames_path = os.path.join(folder_path, "frames")
    if (not os.path.exists(frames_path)) or overwrite:
        folder_to_frames(folder_path, flip_channels=["rgb"], rotate=True, exclude=["800", "975"])
    # align_folder(frames_path, plot_res=False, zed_roi_params=dict(x_s=0, x_e=1080, y_s=310, y_e=1670), zed_shift=0)
    # folder_alignmnet_df = pd.read_csv(os.path.join(frames_path, "jai_cors_in_zed.csv"))
    # frame_ids = folder_alignmnet_df["frame"]
    # z_shifts = folder_alignmnet_df["zed_shift"].astype(int)
    x1, y1, x2, y2 = (25, 345, 965, 1635)
    sx = w / (x2-x1)
    sy = h / (y2-y1)
    M = np.array([[5.91707032e-01,  5.78708316e-02, -1.32766709e+02],
       [1.73415604e-02,  6.28651712e-01,  2.98266973e+02]])
    frames = [frame.split(".")[0].split("_")[-1] for frame in os.listdir(frames_path) if "FSI" in frame]
    frames.sort(key=lambda x: int(x))
    zed_shift = -33
    if process:
        for frame in frames[100:250]:
            # if frame > 160:
            #     x1, y1, x2, y2 = (25, 345, 965, 1635)
            #     sx = w / (x2 - x1)
            #     sy = h / (y2 - y1)
            print(f"\r frame: {frame}", end="")
            zed_frame = int(frame)+zed_shift
            jai_im_path = f"/media/yotam/easystore/preTiltDown40cm/relevent_movies/frames/channel_FSI_frame_{frame}.jpg"
            zed_im_path = f"/media/yotam/easystore/preTiltDown40cm/relevent_movies/frames/frame_{zed_frame}.jpg"
            zed = cv2.imread(zed_im_path)[:, :, ::-1]
            jai = cv2.imread(jai_im_path)[:, :, ::-1]
            measurments = pd.read_csv(meas_path)
            sub_meas = measurments[measurments["frame"] == zed_frame]
            if len(sub_meas)>0:
                for i, row in sub_meas.iterrows():
                    track_id = int(row["track_id"])
                    bx1, bx2, by1, by2 = int(row["x1"]), int(row["x2"]), int(row["y1"]), int(row["y2"])
                    zed = cv2.rectangle(zed.astype(np.uint8), (bx1, by1), (bx2, by2), color=(0, 0, 0), thickness=3)
                    # jai = cv2.rectangle(jai.astype(np.uint8), (int((bx1-x1)*sx), int((by1-y1)*sy)), (int((bx2-x1)*sx), int((by2-y1)*sy)),
                    #                     color=(0, 0, 0), thickness=3)
                    zed = draw_dets(zed.astype(np.uint8),track_id, (bx1, by1, bx2, by2), scale=10)
                    jai = draw_dets(jai.astype(np.uint8), track_id, (int((bx1-x1)*sx), int((by1-y1)*sy), int((bx2-x1)*sx), int((by2-y1)*sy)))
            resized_jai, resized_zed = cv2.resize(jai.copy(), (768, 1024)), cv2.resize(zed.copy(), (768, 1024))
            comb_pic = np.zeros((1024, 1536, 3))
            comb_pic[:, :768] = resized_zed
            comb_pic[:, 768:] = resized_jai
            cv2.imwrite(f"/media/yotam/easystore/preTiltDown40cm/relevent_movies/jai_for_video/jai_{zed_frame}.jpg", jai[:, :, ::-1])
            cv2.imwrite(f"/media/yotam/easystore/preTiltDown40cm/relevent_movies/zed_for_video/zed_{zed_frame}.jpg", zed[:, :, ::-1])
            cv2.imwrite(f"/media/yotam/easystore/preTiltDown40cm/relevent_movies/comb_for_video/comb_{zed_frame}.jpg", comb_pic[:, :, ::-1])
            save_to = os.path.join(folder_path, "with_dets", f"frame_{zed_frame}.png")
            plot_2_imgs(zed[y1:y2, x1:x2], jai, title=frame, save_to=save_to, save_only=True)
    frames_to_video("/media/yotam/easystore/preTiltDown40cm/relevent_movies/jai_for_video",
                    "/media/yotam/easystore/preTiltDown40cm/jai_with_dets.mp4", frame_size=(1536, 2048))
    frames_to_video("/media/yotam/easystore/preTiltDown40cm/relevent_movies/jai_for_video",
                    "/media/yotam/easystore/preTiltDown40cm/jai_with_dets_rev.mp4", frame_size=(2048, 1536))
    frames_to_video("/media/yotam/easystore/preTiltDown40cm/relevent_movies/zed_for_video",
                    "/media/yotam/easystore/preTiltDown40cm/zed_with_dets.mp4", frame_size=(1080, 1920))
    frames_to_video("/media/yotam/easystore/preTiltDown40cm/relevent_movies/comb_for_video",
                    "/media/yotam/easystore/preTiltDown40cm/comb_with_dets.mp4", frame_size=(1536, 1024))
