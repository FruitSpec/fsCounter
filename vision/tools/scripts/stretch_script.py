import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from vision.tools.video_wrapper import video_wrapper
from vision.misc.help_func import validate_output_path


def analyze_frame(c_800, c_975, frame_blue, output_folder):

    plt.figure()
    plt.hist(c_800.flatten(), 255)
    output_path_800 = os.path.join(output_folder, 'hist_800.jpg')
    plt.savefig(output_path_800)

    plt.figure()
    plt.hist(c_975.flatten(), 255)
    output_path_975 = os.path.join(output_folder, 'hist_975.jpg')
    plt.savefig(output_path_975)

    plt.figure()
    plt.hist(frame_blue.flatten(), 255)
    output_path_blue = os.path.join(output_folder, 'hist_blue.jpg')
    plt.savefig(output_path_blue)

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

def clahe(frame_800, frame_975, frame_blue):
    clahe = cv2.createCLAHE(clipLimit=15., tileGridSize=(17, 17))
    clahe_800 = cv2.createCLAHE(clipLimit=2., tileGridSize=(17, 17))
    fsi = np.zeros((frame_800.shape[0], frame_800.shape[1], 3), dtype=np.uint8)

    # norm_800 = np.zeros((frame_800.shape[0], frame_800.shape[1]))
    # norm_800 = cv2.normalize(frame_800, norm_800, 0, 255, cv2.NORM_MINMAX)
    norm_800 = frame_800

    #norm_975 = np.zeros((frame_975.shape[0], frame_975.shape[1]))
    #norm_975 = cv2.normalize(frame_975, norm_975, 0, 255, cv2.NORM_MINMAX)
    norm_975 = frame_975


    g_800 = clahe_800.apply(norm_800)
    #g_975 = cv2.equalizeHist(norm_975)
    g_975 = clahe.apply(norm_975)

    save_hist(g_800, output_folder, 'hist_800_post.jpg')
    save_hist(g_975, output_folder, 'hist_975_post.jpg')

    #frame_blue[frame_blue > 100] = 100

    fsi[:, :, 0] = g_800
    fsi[:, :, 1] = g_975
    fsi[:, :, 2] = frame_blue

    return fsi

def get_fsi(frame_800, frame_975, frame_blue, output_folder=None, stretch_blue=False,norm=True, eq=True):

    fsi = np.zeros((frame_800.shape[0], frame_800.shape[1], 3), dtype=np.uint8)
    if norm:
        norm_800 = np.zeros((frame_800.shape[0], frame_800.shape[1]))
        norm_800 = cv2.normalize(frame_800, norm_800, 0, 255, cv2.NORM_MINMAX)

        norm_975 = np.zeros((frame_975.shape[0], frame_975.shape[1]))
        norm_975 = cv2.normalize(frame_975, norm_975, 0, 255, cv2.NORM_MINMAX)
    else:
        norm_800 = frame_800
        norm_975 = frame_975

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


def create_compare_clip(row_folder, output_path, max_id=200):
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

            fsi = clahe(frame_800[:, :, 0], frame_975[:, :, 0], frame_rgb[:, :, 2])
            fsi = cv2.cvtColor(fsi, cv2.COLOR_RGB2BGR)

            fsi = cv2.resize(fsi, (480, 640))
            frame_fsi = cv2.resize(frame_fsi, (480, 640))
            frame = np.hstack([frame_fsi, fsi])
            output_video.write(frame)
            f_id += 1


    output_video.release()
    print('done')


if __name__ == "__main__":

    #row_folder = r'C:\Users\Matan\Documents\fsi\frames_set1\row_4\1'
    row_folder = r'C:\Users\Matan\Documents\fsi\frames_default_shade\row_3\1'
    #row_folder = r'C:\Users\Matan\Documents\fsi\frames_default\row_2\1'
    output_folder = r'C:\Users\Matan\Documents\fsi\raw\analysis_output_def_shape_clahe_t14'
    validate_output_path(output_folder)
    #create_clip(row_folder, output_folder)
    #create_compare_clip(row_folder, output_folder)

    f_id = 50

    frame_800, frame_975, frame_rgb, frame_fsi = get_frames(row_folder, f_id)
    frame_800 = frame_800[:, :, 0].copy()
    frame_975 = frame_975[:, :, 0].copy()
    frame_blue = frame_rgb[:, :, 2].copy()
    analyze_frame(frame_800, frame_975, frame_blue, output_folder)
    #fsi_blue = get_fsi(frame_800, frame_975, frame_rgb[:, :, 2], output_folder, stretch_blue=True)
    #plt.imsave(os.path.join(output_folder, f'fsi_blue_{f_id}.jpg'), fsi_blue)
    #
    #frame_blue[frame_blue > 200] = 200
    #
    #fsi = get_fsi(frame_800, frame_975, frame_blue, output_folder, stretch_blue=False, norm=False, eq=True)
    #plt.imsave(os.path.join(output_folder, f'fsi_{f_id}.jpg'), fsi)
    #
    # #fsi_at = adaptive_threshold(frame_800, frame_975, frame_rgb)
    # #plt.imsave(os.path.join(output_folder, f'fsi_at_{f_id}.jpg'), fsi_at)
    #
    fsi_g = clahe(frame_800, frame_975, frame_rgb[:, :, 2])
    plt.imsave(os.path.join(output_folder, f'fsi_clahe_{f_id}.jpg'), fsi_g)
    #
    # #fsi_g = gamma(frame_800, frame_975, frame_rgb[:, :, 2])
    # #plt.imsave(os.path.join(output_folder, f'fsi_gamma_{f_id}.jpg'), fsi_g)
    #
    cv2.imwrite(os.path.join(output_folder, f'frame_fsi_{f_id}.jpg'), frame_fsi)
