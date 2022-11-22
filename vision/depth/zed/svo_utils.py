import os

import pyzed.sl as sl
import numpy as np
import cv2
import random
import string

from vision.depth.zed.clip_depth_viewer import init_cam

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

def svo_to_frames(filepath, output_path_name, max_frame=None, rotate=False):

    counter = 0
    cam, runtime = init_cam(filepath)
    if max_frame is None:
        max_frame = cam.get_svo_number_of_frames()

    mat = sl.Mat()
    depth = sl.Mat()
    xyz = sl.Mat()
    key = ''
    while True:  # for 'q' key
        err = cam.grab(runtime)
        if err == sl.ERROR_CODE.SUCCESS and counter < max_frame:
            cam_run_p = cam.get_init_parameters()
            cam.retrieve_image(mat, sl.VIEW.LEFT)
            generated_path = os.path.join(output_path_name, f"frame_{counter}.jpg")
            image = mat.get_data()
            if rotate:
                image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            cv2.imwrite(generated_path, image[:, :, :3])

            cam.retrieve_measure(depth, sl.MEASURE.DEPTH)
            cam.retrieve_measure(xyz, sl.MEASURE.XYZRGBA)
            depth_img = depth.get_data()
            xyz_img = xyz.get_data()[:,:,:3]
            depth_img = (cam_run_p.depth_maximum_distance - np.clip(depth_img, 0,
                                                                    cam_run_p.depth_maximum_distance)) * 255 / cam_run_p.depth_maximum_distance
            bool_mask = np.where(np.isnan(depth_img), True, False)
            depth_img[bool_mask] = 0
            # if remove_high_blues:
            if True:
                mat = sl.Mat()
                cam.retrieve_image(mat, sl.VIEW.LEFT)
                depth_img[mat.get_data()[:, :, 0] > 190] = 0
            depth_img = cv2.medianBlur(depth_img, 5)
            if rotate:
                depth_img = cv2.rotate(depth_img, cv2.ROTATE_90_CLOCKWISE)
                xyz_img = cv2.rotate(xyz_img, cv2.ROTATE_90_CLOCKWISE)
            generated_path = os.path.join(output_path_name, f"depth_frame_{counter}.jpg")
            generated_xyz_path = os.path.join(output_path_name, f"xyz_frame_{counter}.npy")
            cv2.imwrite(generated_path, depth_img)
            np.save(os.path.join(output_path_name, f"xyz_frame_{counter}.npy"), xyz_img)

            counter += 1
        else:
            break
    cam.close()


if __name__ == "__main__":
    fp ="/home/fruitspec-lab/FruitSpec/Sandbox/merge_sensors/HD720_SN39577186_M_11-06-35.svo"
    output_path = "/home/fruitspec-lab/FruitSpec/Sandbox/merge_sensors/ZED2"
    svo_to_frames(fp, output_path, 600, True)