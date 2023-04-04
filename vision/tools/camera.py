import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.exposure import adjust_gamma



def is_sturated(img, percentile=0.8, threshold=245):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, b = np.histogram(gray.flatten(), 255)
    norm = np.cumsum(h) / np.sum(h)
    ind = np.argmax(norm >= percentile)

    return b[ind] >= threshold

def fsi_from_channels(rgb, c_800, c_975, lower=0.005, upper=0.995, min_int=25, max_int=235):
    clahe = cv2.createCLAHE(2, (10, 10))
    rgb = stretch_rgb(rgb, lower, upper, min_int, max_int, clahe)
    c_800 = stretch_and_clahe(c_800[:, :, 0], lower, upper, min_int, max_int, clahe)
    c_975 = stretch_and_clahe(c_975[:, :, 0], lower, upper, min_int, max_int, clahe)

    fsi = rgb.copy()
    fsi[:, :, 0] = c_800.copy()
    fsi[:, :, 1] = c_975.copy()

    return fsi, rgb

def stretch_and_clahe(channel, lower, upper, min_int, max_int, clahe):
    channel = stretch(channel, lower, upper, min_int, max_int)
    channel = clahe.apply(channel)

    return channel


def stretch_rgb(rgb, lower, upper, min_int, max_int, clahe):
    out = rgb.copy()
    out[:, :, 0] = stretch_and_clahe(rgb[:, :, 0].copy(), lower, upper, min_int, max_int, clahe)
    out[:, :, 1] = stretch_and_clahe(rgb[:, :, 1].copy(), lower, upper, min_int, max_int, clahe)
    out[:, :, 2] = stretch_and_clahe(rgb[:, :, 2].copy(), lower, upper, min_int, max_int, clahe)

    return out


def stretch(img, lower, upper, min_int, max_int):

    normalized_img = (img.astype(np.float32) - img.min()) / (img.max() - img.min())
    h, b = np.histogram(normalized_img.flatten(), 255)
    total = np.sum(h)
    accumulated = np.cumsum(h).astype(np.float32) / total

    for i, h_ in enumerate(accumulated):
        if h_ >= lower:
            break
    lower_threshold = b[i]

    for i in range(len(accumulated) - 1, 0, -1):
        if accumulated[i] <= upper:
            break
    upper_threshold = b[i]

    gain = (max_int - min_int) / upper_threshold

    offset = min_int

    stretched_img = normalized_img * gain + offset
    stretched_img = np.clip(stretched_img, 0, 255)

    type_ = np.uint8
    return stretched_img.astype(type_)
def jai_to_channels(jai_frame):

    rgb = cv2.demosaicing(jai_frame[:,:,0], cv2.COLOR_BAYER_BG2RGB)
    channel_1 = jai_frame[:,:,1].copy()
    channel_2 = jai_frame[:, :, 2].copy()

    return rgb, channel_1, channel_2

def old_fsi(rgb, r_ch, g_ch):
    r_ch = stretch_img(r_ch, 255, 0)
    g_ch = stretch_img(g_ch, 255, 0)

    res = rgb.copy()
    res[:, :, 0] = r_ch
    res[:, :, 1] = g_ch

    return res.astype(np.uint8)


def generate_fsi(rgb, r_ch, g_ch):

    g_ch = reduce_outliers(g_ch, 0, 0.005)
    r_ch = reduce_outliers(r_ch, 0, 0.005)

    diff = r_ch.astype(np.int32) - g_ch.astype(np.int32)

    g_ch = stretch_img(g_ch, r_ch.max() - r_ch.min(), r_ch.min())
    ndri = diff / g_ch

    ndri_ch = stretch_img(ndri, 255, 0)
    g_ch = stretch_img(g_ch, 255, 0)

    res = rgb.copy()
    res[:, :, 0] = ndri_ch
    res[:, :, 1] = g_ch

    return res.astype(np.uint8)


def generate_fsi_2(rgb, r_ch, g_ch, gamma=4/5):

    lower_target_intensity = 20
    upper_target_intensity = 235
    #g_ch = reduce_outliers(g_ch, 0, 0.005)
    #r_ch = reduce_outliers(r_ch, 0, 0.005)

    diff = r_ch.astype(np.int32) - g_ch.astype(np.int32)

    g_upper, g_lower = find_gl_by_percentile(g_ch, 0.95, 0.05)

    gain = (upper_target_intensity - lower_target_intensity) / (g_upper - g_lower)
    #offset =
    g_ch = stretch_img(g_ch, r_ch.max() - r_ch.min(), r_ch.min())
    ndri = diff / (g_ch +  1E-5)

    ndri_ch = stretch_img(ndri, 255, 0)
    r_ch = adjust_gamma(r_ch, gamma)
    r_ch = stretch_img(r_ch, 255, 0)

    res = rgb.copy()
    res[:, :, 0] = ndri_ch
    res[:, :, 1] = r_ch

    return res.astype(np.uint8)


def get_file_list(folder_path):
    temp_list = os.listdir(folder_path)
    final_list = []
    for file in temp_list:
        temp = file.split('.')[-1]
        if temp == 'jpg' or temp == 'JPG' or temp == 'png':
            final_list.append(file)

    return final_list


def load_img(file_path):
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def plot_img(img, cmap=None):
    f, ax = plt.subplots(1, 1, figsize=(15, 10))
    if cmap is None:
        ax.imshow(img)
    else:
        ax.imshow(img, cmap)


def plot_two_img(img1, img2):
    f, ax = plt.subplots(1, 2, figsize=(15, 7))
    ax[0].imshow(img1)
    ax[1].imshow(img2)


def preprocess(img, threshold=10):
    r_ch = img[:, :, 0].copy()
    g_ch = img[:, :, 1].copy()

    diff = r_ch.astype(np.float16) - g_ch.astype(np.float16)
    ndvi = diff / (g_ch + 1E-5)

    # diff = img[:,:,0].astype(np.int16) - img[:,:,1].astype(np.int16)
    # diff = remove_outliers(diff)
    # diff[diff < threshold] = 0
    g_ch = reduce_outliers(g_ch, 0, 0.002)
    g = (g_ch.astype(np.float16) - g_ch.min()) / (g_ch.max() - g_ch.min()) * 255
    d = (ndvi.astype(np.float16) - ndvi.min()) / (ndvi.max() - ndvi.min()) * 255
    res = img.copy()
    res[:, :, 0] = d.astype(np.uint8)
    res[:, :, 1] = g.astype(np.uint8)

    return res


def remove_outliers(diff, lower_percentile=0.05, upper_percentile=0.0005):
    upper_threshold, lower_threshold = find_gl_by_percentile(diff, lower_percentile, upper_percentile)

    res = diff.copy()
    res[res >= upper_threshold] = 0
    res[res <= lower_threshold] = 0

    return res


def find_gl_by_percentile(channel, upper, lower):
    h, b = np.histogram(channel.flatten(), 256)
    total = np.sum(h)
    accumulated = np.cumsum(h).astype(np.float32) / total

    for i, h_ in enumerate(accumulated):
        if h_ >= lower:
            break
    lower_intensity = b[i]

    for i in range(len(accumulated) - 1, 0, -1):
        if accumulated[i] <= 1 - upper:
            break
    upper_intensity = b[i]

    return upper_intensity, lower_intensity


def reduce_outliers(diff, lower_percentile=0.05, upper_percentile=0.0005):
    h, b = np.histogram(diff.flatten(), 256)
    total = np.sum(h)
    accumulated = np.cumsum(h).astype(np.float32) / total

    for i, h_ in enumerate(accumulated):
        if h_ >= lower_percentile:
            break
    lower_threshold = b[i]

    for i in range(len(accumulated) - 1, 0, -1):
        if accumulated[i] <= 1 - upper_percentile:
            break
    upper_threshold = b[i]
    min_value = diff.min()

    res = diff.copy()
    res[res >= upper_threshold] = upper_threshold
    res[res <= lower_threshold] = min_value

    return res


def calculate_2dft(input):
    ft = np.fft.ifftshift(input)
    ft = np.fft.fft2(ft)
    return np.fft.fftshift(ft)


def stretch_img(img, gain, offset):

    normalized_img = (img.astype(np.float32) - img.min()) / (img.max() - img.min())

    stretched_img = normalized_img * gain + offset

    if gain > 2**8 - 1:
        type_ = np.uint16
    else:
        type_ = np.uint8

    return stretched_img.astype(type_)

if __name__ == "__main__":
    #fp = r'C:\Users\Matan\Documents\Projects\Data\wetransfer_jai_samples_2022-08-11_0745\JAI_Samples\S4'
    #r_ch = np.array(Image.open(os.path.join(fp, 'Stream1_1089.tiff')))
    #g_ch = np.array(Image.open(os.path.join(fp, 'Stream2_1089.tiff')))
    #rgb = np.array(Image.open(os.path.join(fp, 'Stream0_1089.tiff')))
    fp = '/home/fruitspec-lab/FruitSpec/Data/JAI_FSI_V6_COCO/val2017'
    img_list = os.listdir(fp)
    img = cv2.imread(os.path.join(fp,img_list[0]))
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    r_ch = rgb[:,:,0].copy()
    g_ch = rgb[:, :, 1].copy()

    #g_ch = reduce_outliers(g_ch, 0, 0.005)
    #r_ch = reduce_outliers(r_ch, 0, 0.005)

    diff = r_ch.astype(np.float32) - g_ch.astype(np.float32)
    sum_ = r_ch.astype(np.float32) + g_ch.astype(np.float32)

    g_ch = stretch_img(g_ch, r_ch.max() - r_ch.min(), 10)
    ndri = diff / g_ch

    ndri_ch = stretch_img(ndri, 255, 0)
    g_ch = stretch_img(g_ch, 255, 0)

    res = rgb.copy()
    res[:,:, 0] = ndri_ch
    res[:,:, 1] = g_ch


    plt.imshow(res)
    plt.show()

    a=1