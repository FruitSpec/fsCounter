import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from skimage.exposure import adjust_gamma

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


def generate_fsi_2(rgb, r_ch, g_ch):

    g_ch = reduce_outliers(g_ch, 0, 0.005)
    r_ch = reduce_outliers(r_ch, 0, 0.005)

    diff = r_ch.astype(np.int32) - g_ch.astype(np.int32)

    g_ch = stretch_img(g_ch, r_ch.max() - r_ch.min(), r_ch.min())
    ndri = diff / g_ch

    ndri_ch = stretch_img(ndri, 255, 0)
    ndri_ch = adjust_gamma(ndri_ch, 1 / 2)
    g_ch = stretch_img(g_ch, 255, 0)

    res = rgb.copy()
    res[:, :, 0] = ndri_ch
    res[:, :, 1] = g_ch

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

    res = diff.copy()
    res[res >= upper_threshold] = 0
    res[res <= lower_threshold] = 0

    return res


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
    print(img.max() - img.min())
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