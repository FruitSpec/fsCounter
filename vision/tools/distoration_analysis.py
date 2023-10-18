import numpy as np
import cv2
import matplotlib.pyplot as plt
from os import listdir,path
from tqdm import tqdm
import glob
from sensors_alignment import affine_to_values


def show_img(img, cmap=""):
    if cmap == "":
        plt.imshow(img)
    else:
        plt.imshow(img, cmap=cmap)
    plt.show()


def plot_2_imgs(img1, img2, title=""):
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
        return gray, img[:, :, ::-1]
    return gray


def findChessboardCorners_wresize(gray, n_cols, n_rows):
    ret, corners = cv2.findChessboardCorners(gray, (n_cols - 1, n_rows - 1), None)
    if not ret:
        smaller_size_50 = (int(gray.shape[1] * 0.5), int(gray.shape[0] * 0.5))
        ret, corners = cv2.findChessboardCorners(cv2.resize(gray, smaller_size_50), (n_cols - 1, n_rows - 1), None)
        if not isinstance(corners, type(None)):
            corners /= 0.5
    return ret, corners


def draw_chess_board(folder_path, n_rows=7, n_cols=10, cut_img=(0, 1920, 0, 1080),
                     show_plots=True, with_valid=False):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    objp, objpoints, imgpoints = init_objp(n_rows-1, n_cols-1)
    if with_valid:
        valid = []
    for image in tqdm(listdir(folder_path)):
        file_path = path.join(folder_path, image)
        gray, img = read_gray(file_path, return_org=True, cut_img=cut_img)
        ret, corners = findChessboardCorners_wresize(gray, n_cols, n_rows)
        if with_valid:
            valid.append(ret)
        if ret:
            objpoints.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)
            img = img.astype(np.uint8)
            if show_plots:
                img = cv2.drawChessboardCorners(img, (n_cols-1, n_rows-1), corners2, ret)
                show_img(img)
                # cv2.imshow('img', img)
                # cv2.waitKey()
        elif with_valid:
            imgpoints.append(objpoints)
            objpoints.append(objpoints)
        if show_plots:
            cv2.destroyAllWindows()
    if with_valid:
        return objpoints, imgpoints, gray, img, valid
    return objpoints, imgpoints, gray, img


def get_corners_for_picture(gray, n_cols=7, n_rows=10):
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    ret, corners = findChessboardCorners_wresize(gray, n_cols, n_rows)
    if ret:
        out_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)[:, 0, :]
    else:
        out_corners = np.array([])
    return out_corners


def chess_board_2_cameras_translation(folder_path_zed, folder_path_jai, n_rows=7, n_cols=10,
                            cut_img_zed=(0, 1080, 0, 1920), cut_img_jai=(0, 1536, 0, 2048)):
    objpoints_jai, imgpoints_jai, _, _, valid_jai = draw_chess_board(folder_path_jai, n_rows, n_cols,
                                                                     cut_img_jai, with_valid=True)
    objpoints_zed, imgpoints_zed, _, _, valid_zed = draw_chess_board(folder_path_zed, n_rows, n_cols,
                                                                     cut_img_zed, with_valid=True)
    valid_pics_both = np.all([valid_jai, valid_zed], axis=0)
    imgpoints_zed = [img for img, ret in zip(imgpoints_zed, valid_pics_both) if ret]
    imgpoints_jai = [img for img, ret in zip(imgpoints_jai, valid_pics_both) if ret]
    M = cv2.estimateAffine2D(np.array(imgpoints_zed).reshape(-1, 1, 2)[:, 0, :],
                             np.array(imgpoints_jai).reshape(-1, 1, 2)[:, 0, :])[0]
    return M


def get_calibration_params(objpoints, imgpoints, gray):
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    return ret, mtx, dist, rvecs, tvecs


def undistord_img(img, mtx, dist, w, h, title=""):
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
    x, y, w, h = roi
    dst = dst[y:y+h, x:x+w]
    plot_2_imgs(img, dst, title="before                  after" if title == "" else title)


def cut_zed_in_jai(zed_img, cur_coords):
    x1 = max(cur_coords["x1"][0], 0)
    x2 = min(cur_coords["x2"][0], zed_img.shape[1])
    y1 = max(cur_coords["y1"][0], 0)
    y2 = min(cur_coords["y2"][0], zed_img.shape[0])
    # x1, x2 = 145, 1045
    # y1, y2 = 370, 1597
    if len(zed_img.shape) > 2:
        zed_img = zed_img[y1:y2, x1:x2, :]
    else:
        zed_img = zed_img[y1:y2, x1:x2]
    return zed_img


def get_coordinates_in_zed(gray_zed, gray_jai, tx, ty, sx, sy):
    """
    converts translation output to bbox (x1, y1, x2, y2)
    :param gray_zed: zed gray image
    :param gray_jai: jai gray image
    :param tx: translation in x axis
    :param ty: translation in y axis
    :param sx: scale in x axis
    :param sy: scale in y axis
    :return: x1, y1, x2, y2 of jai image inside zed
    """
    jai_in_zed_height = np.round(gray_jai.shape[0] * sy).astype(np.int)
    jai_in_zed_width = np.round(gray_jai.shape[1] * sx).astype(np.int)
    z_w = gray_zed.shape[1]
    x1 = tx
    x2 = tx + jai_in_zed_width
    if ty > 0:
        y1 = ty
        y2 = ty + jai_in_zed_height
    else:
        y1 = - ty
        y2 = jai_in_zed_height - ty
    return x1, y1, x2, y2


if __name__ == "__main__":
    # for image in listdir("/home/fruitspec-lab/Documents/ZED"):
    #     print(image)
    #     file_path = path.join("/home/fruitspec-lab/Documents/ZED", image)
    #     if "png" in image:
    #         draw_chess_board(file_path)
    # for i in range(103, 150):
    #     img_zed_path = f"/media/fruitspec-lab/easystore/ch_st/zed_rgb/frame_{i}.jpg"
    #     img_rgb_path = f"/media/fruitspec-lab/easystore/ch_st/jai_rgb/channel_RGB_frame_{i}.jpg"
    #     gray_zed, zed_img = read_gray(img_zed_path, return_org=True)
    #     gray_jai_rgb, jai_rgb = read_gray(img_rgb_path, return_org=True)
    #     zed_chess_board_point = get_corners_for_picture(gray_zed, n_rows=4, n_cols=5)
    #     cut_coords = {"x1": [20], "x2": [940], "y1": [360], "y2": [1620]}
    #     zed_cut = cut_zed_in_jai(zed_img, cut_coords)
    #     # M = np.array([[6.01312693e-01, -2.40858197e-02,  4.15215540e+01],
    #     #               [-2.24936494e-03,  5.92510960e-01,  3.85117129e+02],
    #     #               [0, 0, 1]])
    #     # corners_from_zed_to_jai = [(np.linalg.inv(M) @ [x, y, 1])[:-1].astype(np.int) for x, y in zed_chess_board_point]
    #     zed_cut_gray = cv2.resize(cut_zed_in_jai(gray_zed, cut_coords), (1536, 2048))
    #     corners_from_zed_to_jai = get_corners_for_picture(zed_cut_gray, n_rows=4, n_cols=5).astype(np.int)
    #     for point in corners_from_zed_to_jai:
    #         jai_rgb = cv2.circle(jai_rgb.astype(np.uint8), tuple(point), 5, (255, 0, 0), 3)
    #     show_img(jai_rgb)
    #     plot_2_imgs(zed_cut, jai_rgb)
    # zed_points_in_jai = [[x, y, 1] for x, y in zed_chess_board_point]



    # folder_path_zed = "/media/fruitspec-lab/easystore/ch_st/zed_rgb"
    # folder_path_jai = "/media/fruitspec-lab/easystore/ch_st/jai_rgb"
    folder_path_zed = "/media/fruitspec-lab/TEMP SSD/calibration_83/chess/row_4/1/frames_ZED"
    folder_path_jai = "/media/fruitspec-lab/TEMP SSD/calibration_83/chess/row_4/1/frames_RGB"
    # M = chess_board_2_cameras_translation(folder_path_jai, folder_path_zed, n_rows=9, n_cols=6,
    #                         cut_img_zed=(0, 2048-180, 265, 1285), cut_img_jai=(0, 1920, 0, 1080))
    # tx, ty, sx, sy = affine_to_values(M) # (29, 379, 0.5988012493806433, 0.5985438828810826) ## (-255, 384, 0.6028659991474947, 0.598818198428677)
    # x1,y1,x2,y2 = (29, 379, 949, 1605) ###### (-101, 384, 825, 1610)
    folder_path = folder_path_jai
    objpoints, imgpoints, gray, img = draw_chess_board(folder_path, n_rows=9, n_cols=6, cut_img=(0,2048,0,1536))
    ret, mtx, dist, rvecs, tvecs = get_calibration_params(objpoints, imgpoints, gray)
    w, h = img.shape[1], img.shape[0]
    for image in listdir(folder_path)[:5]:
        file_path = path.join(folder_path, image)
        img = cv2.imread(file_path) # [:1080, :1920]
        undistord_img(img, mtx, dist, w, h, title=image)
    print("done")