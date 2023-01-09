import os
import cv2
import numpy as np
from tqdm import tqdm

from vision.tools.sensors_alignment import align_sensors
from vision.tools.image_stitching import resize_img

def write_video(zed_folder, jai_folder, zed_frames, jai_frames, output_path):

    zed_frame_list = []
    depth_frame_list = []
    for i in zed_frames:
        zed_frame_list.append(os.path.join(zed_folder, f"frame_{i}.jpg"))
        depth_frame_list.append(os.path.join(zed_folder, f"depth_frame_{i}.jpg"))

    rgb_jai_list = []
    fsi_jai_list = []
    for i in jai_frames:
        fsi_jai_list.append(os.path.join(jai_folder, f"frame_{i}.jpg"))
        rgb_jai_list.append(os.path.join(jai_folder, f"rgb_{i}.jpg"))

    movie_path = os.path.join(output_path, 'zed_jai.avi')
    zed_jai = cv2.VideoWriter(movie_path, cv2.VideoWriter_fourcc('M','J','P','G'), 5, (750 * 2, 1000))
    movie_path = os.path.join(output_path, 'depth_jai.avi')
    #fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    #depth_jai = cv2.VideoWriter(movie_path, fourcc, 5, (750 * 2, 1000))
    depth_jai = cv2.VideoWriter(movie_path, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 5, (750 * 2, 1000))

    for i in tqdm(range(len(zed_frame_list))):
        zed = cv2.imread(zed_frame_list[i])
        #zed = cv2.rotate(zed, cv2.ROTATE_90_CLOCKWISE)
        zed = cv2.cvtColor(zed, cv2.COLOR_BGR2RGB)

        depth = cv2.imread(depth_frame_list[i])
        #depth = cv2.rotate(depth, cv2.ROTATE_90_CLOCKWISE)
        #depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
        #depth = (10 - np.clip(depth, 0, 10)) * 255 / 10

        jai = cv2.imread(fsi_jai_list[i])
        jai = cv2.cvtColor(jai, cv2.COLOR_BGR2RGB)

        rgb_jai = cv2.imread(rgb_jai_list[i])
        rgb_jai = cv2.cvtColor(rgb_jai, cv2.COLOR_BGR2RGB)

        corr = align_sensors(zed, rgb_jai)
        corr = list(corr.astype(np.int))

        aligned_zed = zed[corr[1]:corr[3], corr[0]:corr[2], :]
        aligned_depth = depth[corr[1]:corr[3], corr[0]:corr[2]]

        f_depth, r_d = resize_img(aligned_depth, 960)
#        f_depth = np.dstack([f_depth, f_depth, f_depth])
        f_zed, r_z = resize_img(aligned_zed, 960)
        f_zed = cv2.cvtColor(f_zed, cv2.COLOR_RGB2BGR)
        f_jai, r_j = resize_img(jai, 960)
        f_jai = cv2.cvtColor(f_jai, cv2.COLOR_RGB2BGR)

        canvas = np.zeros((1000, 750*2, 3), dtype=np.uint8)
        canvas[20:20 + f_zed.shape[0], 15:15 + f_zed.shape[1], :] = f_zed.copy()
        canvas[20:20 + f_jai.shape[0], 765:765 + f_jai.shape[1], :] = f_jai.copy()

        zed_jai.write(canvas)

        canvas = np.zeros((1000, 750 * 2, 3), dtype=np.uint8)
        canvas[20:20 + f_depth.shape[0], 15:15 + f_depth.shape[1], :] = f_depth.copy()
        canvas[20:20 + f_jai.shape[0], 765:765 + f_jai.shape[1], :] = f_jai.copy()

        depth_jai.write(canvas)

    zed_jai.release()
    depth_jai.release()


if __name__ == "__main__":
    zed_folder = "/home/fruitspec-lab/FruitSpec/Sandbox/merge_sensors/ZED2"
    jai_folder = "/home/fruitspec-lab/FruitSpec/Sandbox/merge_sensors/FSI_2_30_720_30"
    #zed_frames = [545, 548, 551, 554, 557, 560, 563, 566, 569, 572, 575, 578, 581] #, 584]
    zed_frames = [547, 548, 549, 550, 551, 552, 553, 554, 555, 556, 557, 558, 559]  # , 584]
    #jai_frames = [539, 541, 543, 545, 547, 549, 551, 553, 555, 557, 559, 561, 563]
    jai_frames = [540, 541, 542, 543, 544, 545, 546, 546, 548, 549, 550, 551, 552]
    output_path = "/home/fruitspec-lab/FruitSpec/Sandbox/merge_sensors/example_3"
    write_video(zed_folder, jai_folder, zed_frames, jai_frames, output_path)