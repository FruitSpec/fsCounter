import numpy as np
import cv2
import matplotlib.pyplot as plt
from os import listdir,path
import glob


def show_img(img, cmap=""):
    if cmap == "":
        plt.imshow(img)
    else:
        plt.imshow(img, cmap=cmap)
    plt.show()

def plot_2_imgs(img1,img2,title=""):
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
    ax1.imshow(img1)
    ax2.imshow(img2)
    plt.title(title)
    plt.show()

def init_objp(n_rows=7, n_cols=10):
    objp = np.zeros((n_rows * n_cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:n_cols, 0:n_rows].T.reshape(-1, 2)
    objpoints, imgpoints = [], []
    return objp, objpoints, imgpoints


def read_gray(file_path, return_org=False, cut_img=None):
    img = cv2.imread(file_path)
    if not isinstance(cut_img, type(None)):
        img = img[cut_img[0]:cut_img[1], cut_img[2]:cut_img[3]]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if return_org:
        return gray, img
    return gray


def draw_chess_board(folder_path, n_rows=7, n_cols=10, cut_img=(0, 1080, 0, 1920), show_plots=False):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp, objpoints, imgpoints = init_objp(n_rows-1, n_cols-1)
    for image in listdir(folder_path):
        file_path = path.join(folder_path, image)
        gray, img = read_gray(file_path, return_org=True, cut_img=cut_img)
        ret, corners = cv2.findChessboardCorners(gray, (n_cols-1, n_rows-1), None)
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            if show_plots:
                cv2.drawChessboardCorners(img, (n_cols-1, n_rows-1), corners2, ret)
                cv2.imshow('img', img)
                cv2.waitKey()
        if show_plots:
            cv2.destroyAllWindows()
    return objpoints, imgpoints, gray, img


def get_calibration_params(objpoints, imgpoints, gray):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return ret, mtx, dist, rvecs, tvecs


def undistord_img(img, mtx, dist, w, h, title=""):
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    plot_2_imgs(img, dst, title="before                  after" if title == "" else title)


if __name__ == "__main__":
    # for image in listdir("/home/fruitspec-lab/Documents/ZED"):
    #     print(image)
    #     file_path = path.join("/home/fruitspec-lab/Documents/ZED", image)
    #     if "png" in image:
    #         draw_chess_board(file_path)

    folder_path = "/home/fruitspec-lab/Documents/ZED/calibaration"
    objpoints, imgpoints, gray, img = draw_chess_board(folder_path)
    ret, mtx, dist, rvecs, tvecs = get_calibration_params(objpoints, imgpoints, gray)
    w, h = img.shape[1], img.shape[0]
    for image in listdir(folder_path):
        file_path = path.join(folder_path, image)
        img = cv2.imread(file_path)[:1080, :1920]
        undistord_img(img, mtx, dist, w, h, title=image)
    print("done")