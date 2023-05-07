import pandas as pd
import cv2
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm

from vision.misc.help_func import get_repo_dir
from vision.tools.video_wrapper import video_wrapper
from vision.tools.sensors_alignment import SensorAligner

def downsample_by_log(log_fp, zed_fp, jai_fp, jai_rgb_fp, output_movie_fp, alignment_output_fp, args):
    sa = SensorAligner(args.sensor_aligner)
    log_df = pd.read_csv(log_fp)
    jai_frame_ids = list(log_df['JAI frame number'])
    zed_frame_ids = list(log_df['ZED frame number'])

    cam_zed = video_wrapper(zed_fp, rotate=2)
    cam_jai = video_wrapper(jai_fp, rotate=1)
    cam_jai_rgb = video_wrapper(jai_rgb_fp, rotate=1)

    alignment_data = []
    f_id = 0

    (x1, y1, x2, y2) = 0, 0, 0, 0
    output_video = cv2.VideoWriter(output_movie_fp, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                   10, (1000, 700), isColor=True)
    jai_first_id, zed_first_id = 0, 0
    p_zed_id = -1e5
    for jai_id, zed_id in tqdm(zip(jai_frame_ids, zed_frame_ids)):
        if p_zed_id > zed_id:
            jai_first_id = jai_id
        p_zed_id = zed_id
        if f_id < 20:
            f_id += 1
            continue
        zed_frame, _, _ = cam_zed.get_zed(zed_id)
        _, jai_frame = cam_jai.get_frame(jai_id - jai_first_id)
        _, jai_rgb_frame = cam_jai_rgb.get_frame(jai_id - jai_first_id)

        (px1, py1, px2, py2) = (x1, y1, x2, y2)
        (x1, y1, x2, y2), tx, ty, kp_zed, kp_jai, gray_zed, gray_jai, match, st = sa.align_calib_sensors(zed_frame, jai_rgb_frame)
        alignment_data.append({'z_shift': sa.zed_shift, 'tx': tx, 'matches': np.sum(st), 'f_id': f_id})

        # winname = f"zed {zed_id} | jai {jai_id}"
        # if np.isnan(x1):
        #     winname = "!!!!!!!!!!!!!!!!!!!!!!"
        #     (x1, y1, x2, y2) = px1, py1, px2, py2
        #
        # canvas = np.zeros((zed_frame.shape[0] + jai_frame.shape[0] + 20, zed_frame.shape[1] + jai_frame.shape[1] + 30, 3), dtype=np.uint8)
        #
        # x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        # print(x1, y1, x2, y2)
        # zed_frame = cv2.rectangle(zed_frame, (x1, y1), (x2, y2), (0, 0, 255), 5)
        #
        # canvas[10:10+zed_frame.shape[0], 10:10+zed_frame.shape[1], :] = zed_frame
        # canvas[10:10+jai_frame.shape[0], 20+zed_frame.shape[1]:20+zed_frame.shape[1]+jai_frame.shape[1], :] = jai_frame
        #
        # cv2.destroyAllWindows()
        # cv2.namedWindow(winname, cv2.WND_PROP_FULLSCREEN)
        # cv2.setWindowProperty(winname, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
        # cv2.resizeWindow(winname, 1000, 1000)
        # cv2.imshow(winname, canvas)
        # cv2.waitKey()

        cropped = zed_frame[int(y1):int(y2), int(x1):int(x2), :]
        cropped = cv2.resize(cropped, (480, 640))

        jai_frame = cv2.resize(jai_frame, (480, 640))

        canvas = np.zeros((700, 1000, 3), dtype=np.uint8)
        canvas[10:10 + cropped.shape[0], 10:10+cropped.shape[1], :] = cropped
        canvas[10:10 + jai_frame.shape[0], 510:510 + jai_frame.shape[1], :] = jai_frame

        f_id += 1
        output_video.write(canvas)

    output_video.release()
    df = pd.DataFrame(alignment_data, columns=['z_shift', 'tx', 'matches', 'f_id'])
    df.to_csv(alignment_output_fp)



if __name__ == "__main__":
    repo_dir = get_repo_dir()
    runtime_config = "/vision/pipelines/config/dual_runtime_config.yaml"
    args = OmegaConf.load(repo_dir + runtime_config)

    zed_fp = "/home/fruitspec-lab/FruitSpec/Data/counter/FPS10_clahe/row_2/ZED_1.svo"
    jai_fp = "/home/fruitspec-lab/FruitSpec/Data/counter/FPS10_clahe/row_2/FSI_CLAHE_1.mkv"
    jai_rgb_fp = "/home/fruitspec-lab/FruitSpec/Data/counter/FPS10_clahe/row_2/RGB_1.mkv"
    log_fp = "/home/fruitspec-lab/FruitSpec/Data/counter/FPS10_clahe/row_2/jaized_timestamps_1.log"
    alignment_output_fp = "/home/fruitspec-lab/FruitSpec/Data/counter/FPS10_clahe/row_2/alignment.csv"
    output_movie_fp = "/home/fruitspec-lab/FruitSpec/Data/counter/FPS10_clahe/row_2/synced_clip.mkv"

    downsample_by_log(log_fp, zed_fp, jai_fp, jai_rgb_fp, output_movie_fp, alignment_output_fp, args)
