import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

from vision.tools.video_wrapper import video_wrapper
from vision.misc.help_func import validate_output_path


def analyze_frame(c_800, c_975, frame_blue, output_folder):

    plt.figure()
    h, b = np.histogram(c_800.flatten(), 255)
    sat_800 = np.sum(h[250:]) / np.sum(h)
    plt.hist(c_800.flatten(), 255)
    output_path_800 = os.path.join(output_folder, 'hist_800.jpg')
    plt.savefig(output_path_800)
    plt.close()

    plt.hist(c_800[frame_blue < 200].flatten(), 255)
    output_path_800 = os.path.join(output_folder, 'hist_no_sky_800.jpg')
    plt.savefig(output_path_800)
    plt.close()



    plt.figure()
    h, b = np.histogram(c_975.flatten(), 255)
    sat_975 = np.sum(h[250:]) / np.sum(h)
    plt.hist(c_975.flatten(), 255)
    output_path_975 = os.path.join(output_folder, 'hist_975.jpg')
    plt.savefig(output_path_975)
    plt.close()

    plt.hist(c_975[frame_blue < 200].flatten(), 255)
    output_path_975 = os.path.join(output_folder, 'hist_no_sky_975.jpg')
    plt.savefig(output_path_975)
    plt.close()

    plt.figure()
    plt.hist(frame_blue.flatten(), 255)
    output_path_blue = os.path.join(output_folder, 'hist_blue.jpg')
    plt.savefig(output_path_blue)
    plt.close()

    return sat_800, sat_975

def get_video_frames(row_folder, f_id):

    cam_975 = video_wrapper(os.path.join(row_folder, 'Result_975.mkv'), 1)
    cam_800 = video_wrapper(os.path.join(row_folder, 'Result_800.mkv'), 1)
    cam_rgb = video_wrapper(os.path.join(row_folder, 'Result_RGB.mkv'), 1)

    _, frame_975 = cam_975.get_frame(f_id)
    _, frame_800 = cam_800.get_frame(f_id)
    _, frame_rgb = cam_rgb.get_frame(f_id)

    return frame_800, frame_975, frame_rgb

def get_frames(row_folder, f_id):

    cam_rgb = video_wrapper(os.path.join(row_folder, 'Result_RGB.mkv'), 1)
    cam_fsi = video_wrapper(os.path.join(row_folder, 'Result_FSI.mkv'), 1)

    frame_975 = cv2.imread(os.path.join(row_folder, f'channel_975_{f_id}.jpg'))
    frame_975 = cv2.rotate(frame_975, cv2.ROTATE_90_COUNTERCLOCKWISE)
    frame_800 = cv2.imread(os.path.join(row_folder, f'channel_800_{f_id}.jpg'))
    frame_800 = cv2.rotate(frame_800, cv2.ROTATE_90_COUNTERCLOCKWISE)
    _, frame_rgb = cam_rgb.get_frame(f_id)
    _, frame_fsi = cam_fsi.get_frame(f_id)

    return frame_800, frame_975, frame_rgb, frame_fsi


def adaptive_threshold(frame_800, frame_975, frame_rgb):
    fsi = np.zeros((frame_800.shape[0], frame_800.shape[1], 3), dtype=np.uint8)
    at_800 = cv2.adaptiveThreshold(frame_800, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 33, 2)
    at_975 = cv2.adaptiveThreshold(frame_975, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 33, 2)
    save_hist(at_800, output_folder, 'postprocess_800_hist.jpg')
    save_hist(at_975, output_folder, 'postprocess_975_hist.jpg')

    fsi[:, :, 0] = at_800
    fsi[:, :, 1] = at_975
    fsi[:, :, 2] = frame_rgb[:, :, 2]

    return fsi

def gamma(frame_800, frame_975, frame_blue):
    fsi = np.zeros((frame_800.shape[0], frame_800.shape[1], 3), dtype=np.uint8)
    gamma = 1.5
    g_800 = cv2.LUT(frame_800, np.power(frame_800 / 255.0, gamma) * 255)
    g_975 = cv2.LUT(frame_975, np.power(frame_975 / 255.0, gamma) * 255)

    fsi[:, :, 0] = g_800
    fsi[:, :, 1] = g_975
    fsi[:, :, 2] = frame_blue

    return fsi

def clahe(frame_800, frame_975, frame_blue, clahe_975=15., clahe_800=5.):
    clahe = cv2.createCLAHE(clipLimit=clahe_975, tileGridSize=(17, 17))
    clahe_800 = cv2.createCLAHE(clipLimit=clahe_800, tileGridSize=(17, 17))
    fsi = np.zeros((frame_800.shape[0], frame_800.shape[1], 3), dtype=np.uint8)

    norm_800 = frame_800
    norm_975 = frame_975

    #norm_800 = np.zeros((frame_800.shape[0], frame_800.shape[1]))
    #norm_800 = cv2.normalize(frame_800, norm_800, 0, 255, cv2.NORM_MINMAX)
    g_800 = clahe_800.apply(norm_800)
    #g_800 = adjust_gamma(norm_800, 1.5)
    #g_975 = adjust_gamma(norm_975, 3)
    g_975 = clahe.apply(norm_975)
    #g_975 = cv2.equalizeHist(norm_975)



    #frame_blue[frame_blue > 100] = 100

    fsi[:, :, 0] = g_800
    fsi[:, :, 1] = g_975
    fsi[:, :, 2] = frame_blue

    return fsi

def get_fsi(frame_800, frame_975, frame_blue, output_folder=None, stretch_blue=False,norm=True, eq=True, diff=False):

    fsi = np.zeros((frame_800.shape[0], frame_800.shape[1], 3), dtype=np.uint8)
    if norm:
        norm_800 = np.zeros((frame_800.shape[0], frame_800.shape[1]))
        norm_800 = cv2.normalize(frame_800, norm_800, 0, 255, cv2.NORM_MINMAX)

        norm_975 = np.zeros((frame_975.shape[0], frame_975.shape[1]))
        norm_975 = cv2.normalize(frame_975, norm_975, 0, 255, cv2.NORM_MINMAX)
    else:
        norm_800 = frame_800
        norm_975 = frame_975
    if diff:
        norm_800 = np.array(norm_800 - (norm_975 * (1/2)), dtype=np.uint8)
        frame_blue = frame_blue // 2
    if eq:
        eq_800 = cv2.equalizeHist(norm_800)
        eq_975 = cv2.equalizeHist(norm_975)
    else:
        eq_800 = norm_800
        eq_975 = norm_975

    save_hist(eq_800, output_folder, 'postprocess_800_hist.jpg')
    save_hist(eq_975, output_folder, 'postprocess_975_hist.jpg')

    if stretch_blue:
        if norm:
            norm_blue = np.zeros((frame_blue.shape[0], frame_975.shape[1]))
            norm_blue = cv2.normalize(frame_blue, norm_blue, 0, 255, cv2.NORM_MINMAX)
        else:
            norm_blue = frame_blue
        if eq:
            eq_blue = cv2.equalizeHist(norm_blue)
        else:
            eq_blue = norm_blue
        save_hist(eq_blue, output_folder, 'eq_blue_hist.jpg')
    else:
        eq_blue = frame_blue

    fsi[:, :, 0] = eq_800
    fsi[:, :, 1] = eq_975
    fsi[:, :, 2] = eq_blue

    return fsi





    if output_folder is not None:
        plt.figure()
        plt.hist(norm_800.flatten(), 255)
        output_path_800 = os.path.join(output_folder, 'norm_800.jpg')
        plt.savefig(output_path_800)


def save_hist(frame, output_path, name=None):
    if name is None:
        name = 'hist.jpg'
    plt.figure()
    plt.hist(frame.flatten(), 255)
    hist_path = os.path.join(output_path, name)
    plt.savefig(hist_path)
    plt.close()

def create_clip(row_folder, output_path, max_id=200):
    file_list = os.listdir(row_folder)
    output_video_name = os.path.join(output_path, 'result_video.avi')
    output_video = cv2.VideoWriter(output_video_name, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                   10, (1536, 2048))

    f_id = 0
    cam_rgb = video_wrapper(os.path.join(row_folder, 'Result_RGB.mkv'), 1)

    while f_id < max_id:
        frame_string = f'channel_800_{f_id}.jpg'
        if frame_string in file_list:
            print(f'frame: {f_id}')
            frame_975 = cv2.imread(os.path.join(row_folder, f'channel_975_{f_id}.jpg'))
            frame_975 = cv2.rotate(frame_975, cv2.ROTATE_90_COUNTERCLOCKWISE)
            frame_800 = cv2.imread(os.path.join(row_folder, f'channel_800_{f_id}.jpg'))
            frame_800 = cv2.rotate(frame_800, cv2.ROTATE_90_COUNTERCLOCKWISE)
            _, frame_rgb = cam_rgb.get_frame()

            fsi = clahe(frame_800[:, :, 0], frame_975[:, :, 0], frame_rgb[:, :, 2])
            fsi = cv2.cvtColor(fsi, cv2.COLOR_RGB2BGR)
            output_video.write(fsi)
            f_id += 1


    output_video.release()
    print('done')


def create_compare_clip(row_folder, output_path, max_id=200, c_975=15, c_800=5):
    file_list = os.listdir(row_folder)
    output_video_name = os.path.join(output_path, 'compare_video.avi')
    output_video = cv2.VideoWriter(output_video_name, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                   10, (960, 640))

    f_id = 0
    cam_rgb = video_wrapper(os.path.join(row_folder, 'Result_RGB.mkv'), 1)
    cam_fsi = video_wrapper(os.path.join(row_folder, 'Result_FSI.mkv'), 1)

    while f_id < max_id:
        frame_string = f'channel_800_{f_id}.jpg'
        if frame_string in file_list:
            print(f'frame: {f_id}')
            frame_975 = cv2.imread(os.path.join(row_folder, f'channel_975_{f_id}.jpg'))
            frame_975 = cv2.rotate(frame_975, cv2.ROTATE_90_COUNTERCLOCKWISE)
            frame_800 = cv2.imread(os.path.join(row_folder, f'channel_800_{f_id}.jpg'))
            frame_800 = cv2.rotate(frame_800, cv2.ROTATE_90_COUNTERCLOCKWISE)
            _, frame_rgb = cam_rgb.get_frame()
            _, frame_fsi = cam_fsi.get_frame()

            fsi = clahe(frame_800[:, :, 0], frame_975[:, :, 0], frame_rgb[:, :, 2], c_975, c_800)
            fsi = cv2.cvtColor(fsi, cv2.COLOR_RGB2BGR)

            fsi = cv2.resize(fsi, (480, 640))
            frame_fsi = cv2.resize(frame_fsi, (480, 640))
            frame = np.hstack([frame_fsi, fsi])
            output_video.write(frame)
            f_id += 1


    output_video.release()
    print('done')

def get_saturated(c):

    h, b = np.histogram(c.flatten(), 255)
    sat = np.sum(h[250:]) / np.sum(h)

    return sat

def analyze_and_stretch(row_folder, output_folder, f_id, c_975=10, c_800=5):
    frame_800, frame_975, frame_rgb, frame_fsi = get_frames(row_folder, f_id)
    frame_800 = frame_800[:, :, 0].copy()
    frame_975 = frame_975[:, :, 0].copy()
    frame_blue = frame_rgb[:, :, 2].copy()
    #sat_800, sat_975 = analyze_frame(frame_800, frame_975, frame_blue, output_folder)
    sat_800, sat_975 = 0, 0
    fsi_no_blue = get_fsi(frame_800,
                          frame_975,
                          frame_blue,
                          output_folder=output_folder,
                          stretch_blue=False,
                          norm=False, eq=True)
    fsi_diff = get_fsi(frame_800,
                          frame_975,
                          frame_blue,
                          output_folder=output_folder,
                          stretch_blue=False,
                          norm=False, eq=True, diff=True)
    plt.imsave(os.path.join(output_folder, f'fsi_no_b_{f_id}.jpg'), fsi_no_blue)
    plt.close()

    plt.imsave(os.path.join(output_folder, f'fsi_diff_{f_id}.jpg'), fsi_diff)
    plt.close()


    fsi_g = clahe(frame_800, frame_975, frame_rgb[:, :, 2], c_975, c_800)
    post_800 = get_saturated(fsi_g[:, :, 0])
    post_975 = get_saturated(fsi_g[:, :, 1])
    save_hist(fsi_g[:, :, 0], output_folder, f'post_clahe_800_{f_id}.jpg')
    save_hist(fsi_g[:, :, 1], output_folder, f'post_clahe_975_{f_id}.jpg')
    plt.imsave(os.path.join(output_folder, f'fsi_clahe_{f_id}.jpg'), fsi_g)
    plt.close()

    cv2.imwrite(os.path.join(output_folder, f'frame_fsi_{f_id}.jpg'), frame_fsi)
    fsi_800 = get_saturated(frame_fsi[:, :, 2]) #BGR
    fsi_975 = get_saturated(frame_fsi[:, :, 1])  # BGR

    return sat_800, sat_975, post_800, post_975, fsi_800, fsi_975

def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)


if __name__ == "__main__":

    #row_folder = r'C:\Users\Matan\Documents\fsi\frames_set1\row_4\1'
    parent_folder = r'C:\Users\Matan\Documents\fsi\alc_and_exp'
    #row_folder = r'C:\Users\Matan\Documents\fsi\alc_and_exp\311023SH1\row_201\1'
    #row_folder = r'C:\Users\Matan\Documents\fsi\frames_default\row_2\1'
    output_folder = r'C:\Users\Matan\Documents\fsi\alc_and_exp\all_rows_analysis_16'
    validate_output_path(output_folder)
    #create_clip(row_folder, output_folder)
    #create_compare_clip(row_folder, output_folder, 15, 5)

    f_id = 31

    steps = os.listdir(parent_folder)
    data = []
    for step in tqdm(steps):
        if '3110' not in step:
            continue
        steps_folder = os.path.join(parent_folder, step)
        rows = os.listdir(steps_folder)
        for row in rows:
            row_folder = os.path.join(steps_folder, row, '1')
            row_output_folder = os.path.join(output_folder, row)
            validate_output_path(row_output_folder)
            sat_800, sat_975, post_800, post_975, fsi_800, fsi_975 = analyze_and_stretch(row_folder,
                                                                                         row_output_folder,
                                                                                         f_id,
                                                                                         c_975=10,
                                                                                         c_800=2)

            data.append({'row': row,
                         'sat_800': sat_800,
                         'sat_975': sat_975,
                         'post_800': post_800,
                         'post_975': post_975,
                         'fsi_800': fsi_800,
                         'fsi_975': fsi_975,
                         })
            print(f'done row {row}')

    columns = list(data[0].keys())
    df = pd.DataFrame(data, columns=columns)
    df.to_csv(os.path.join(output_folder, 'saturation.txt'))