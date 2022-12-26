import os

import pyzed.sl as sl
import numpy as np
import cv2
import random
import string

from vision.depth.zed.clip_depth_viewer import init_cam
from vision.tools.video_wrapper import video_wrapper
from vision.misc.help_func import validate_output_path

def slice_image(filepath, output_path_name, per_svo_sample):
    per_svo_sample = int(per_svo_sample)
    input_type = sl.InputType()
    input_type.set_from_svo_file(filepath)
    init = sl.InitParameters(input_t=input_type, svo_real_time_mode=False)
    cam = sl.Camera()
    status = cam.open(init)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        print(f'file: {filepath}')
        return
    number_of_frames = cam.get_svo_number_of_frames()
    random_ids = get_random_frame_ids(number_of_frames, per_svo_sample)
    runtime = sl.RuntimeParameters()
    mat = sl.Mat()

    f_id = 0
    pbar = tqdm(total=per_svo_sample)
    while True:  # for 'q' key
        err = cam.grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS:
            if f_id in random_ids:
                pbar.update(1)
                cam.retrieve_image(mat)
                generated_name = filepath.split("/")[-1].replace(".svo", "_") + ''.join(random.choices(string.ascii_uppercase + string.ascii_lowercase + string.digits, k=12)) + '.jpg'
                generated_path = os.path.join(output_path_name, generated_name)
                image = mat.get_data()
                cv2.imwrite(generated_path, image[:,:,:3])
            f_id += 1
        else:
            break
    cam.close()

def get_random_frame_ids(number_of_frames, per_svo_sample):
    random_ids = []
    for i in range(per_svo_sample):
        not_found = True
        while not_found:
            id_ = np.random.randint(0, number_of_frames)
            if id_ not in random_ids:
                random_ids.append(id_)
                not_found = False

    return random_ids

def svo_to_frames(filepath, output_path_name, max_frame=None, rotate=0, min_depth=1., max_depth=5.):
    frames_path = os.path.join(output_path_name, "frames")
    validate_output_path(frames_path)
    depth_path = os.path.join(output_path_name, "depth")
    validate_output_path(depth_path)
    counter = 0
    cam = video_wrapper(filepath, rotate, min_depth, max_depth)
    if max_frame is None:
        max_frame = cam.get_number_of_frames()

    while True:  # for 'q' key
        if counter == 163:
            a = 1
        frame, depth, pc = cam.get_zed()
        if not cam.res:
            break
        b = frame[:, :, 0].copy()
        depth[b > 220] = 0
        generated_path = os.path.join(frames_path, f"frame_{counter}.jpg")
        cv2.imwrite(generated_path, frame[:, :, :3])

        generated_path = os.path.join(depth_path, f"depth_frame_{counter}.jpg")
        cv2.imwrite(generated_path, depth)

        counter += 1
        if counter == max_frame:
            break

    cam.close()


if __name__ == "__main__":
    fp ="/home/yotam/FruitSpec/Sandbox/slicer_test/caracara_R2_3011/ZED_1.svo"
    output_path = "/home/yotam/FruitSpec/Sandbox/slicer_test/caracara_R2_3011/sliced3"
    validate_output_path(output_path)
    svo_to_frames(fp, output_path, 300, 2, 1., 3.5)