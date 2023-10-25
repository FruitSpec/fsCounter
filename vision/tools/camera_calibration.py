import os
import numpy as np
import cv2 as cv
import glob
import matplotlib.pyplot as plt
import time


from vision.tools.video_wrapper import video_wrapper
from vision.misc.help_func import validate_output_path


class undistort():

 def __init__(self, calibration_path):

  file = np.load(calibration_path)
  self.mtx = file['mtx']
  self.dist = file['dist']
  self.rvecs = file['rvecs']
  self.tvecs = file['tvecs']

 def apply(self, img):
  return cv.undistort(img, self.mtx, self.dist, None)


def get_calibration_params(objpoints, imgpoints, gray):
 ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
 return ret, mtx, dist, rvecs, tvecs


def plot_2_imgs(img1, img2, title=""):
 fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
 ax1.imshow(img1)
 ax2.imshow(img2)
 plt.title(title)
 plt.show()
def undistord_img(img, mtx, dist, w, h, title=""):
 newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))
 dst = cv.undistort(img, mtx, dist, None, newcameramtx)
 x, y, w, h = roi
 dst = dst[y:y + h, x:x + w]
 plot_2_imgs(img, dst, title="before                  after" if title == "" else title)

def CannyThreshold(src_gray, low_threshold=0, ratio=3, kernel_size=3):

 img_blur = cv.blur(src_gray, (5,5))
 detected_edges = cv.Canny(img_blur, low_threshold, low_threshold*ratio, kernel_size)
 mask = detected_edges != 0
 dst = src_gray * mask
 #cv.imshow('res', dst)

 return dst


def get_corners(objpoints, imgpoints, img, patternSize, objp, resize=False):
 if resize:
  smaller_size_50 = (int(img.shape[1] * 0.5), int(img.shape[0] * 0.5))
  chees_img = cv.resize(img, smaller_size_50)
 else:
  chees_img = img
 ret, corners = cv.findChessboardCorners(chees_img, (patternSize[0], patternSize[1]), None)
 # If found, add object points, image points (after refining them)
 if ret == True:
  objpoints.append(objp)
  # corners2 = cv.cornerSubPix(r_gray,corners, (11,11), (-1,-1), criteria)
  # corners2[:, 0, 0] *= img.shape[1] / smaller_size_50[0]
  # corners2[:, 0, 1] *= img.shape[0] / smaller_size_50[1]

  if resize:
    corners[:, 0, 0] *= img.shape[1] / smaller_size_50[0]
    corners[:, 0, 1] *= img.shape[0] / smaller_size_50[1]
  imgpoints.append(corners)

 return objpoints, imgpoints, corners, ret


def calibrate_lense(video_path, output_path, patternSize = (6,9), start_frame=0, view=False):

 # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
 objp = np.zeros((1, patternSize[0]*patternSize[1], 3), np.float32)
 objp[0, :, :2]  = np.mgrid[0:patternSize[0],0:patternSize[1]].T.reshape(-1, 2)

 # Arrays to store object points and image points from all the images.
 objpoints = [] # 3d point in real world space
 imgpoints = [] # 2d points in image plane.


 rotate = 2 if 'ZED' in video_path else 1
 cam = video_wrapper(video_path, rotate)

 id_ = 0
 ret = True
 while ret:
  id_ += 1
  ret, img = cam.get_frame()

  if id_ < start_frame:
   id_ += 1
   continue

  if ret:
   chees_img = img[:,:,0].copy()
   chees_img[chees_img > 20] = 255
   chees_img[chees_img <= 20] = 0

   print(id_)
   objpoints, imgpoints, corners, ret_c = get_corners(objpoints, imgpoints, chees_img, patternSize, objp, resize=True)
   if ret_c:
     cv.drawChessboardCorners(img, (patternSize[0],patternSize[1]), corners, ret_c)

   if view:
    smaller_size_50 = (int(img.shape[1] * 0.5), int(img.shape[0] * 0.5))
    smaller_img = cv.resize(img, smaller_size_50)
    cv.imshow('img', smaller_img)
    cv.waitKey(50)
   last_image = img

 cv.destroyAllWindows()


 gray = cv.cvtColor(last_image, cv.COLOR_BGR2GRAY)
 print('start calibration')
 ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
 np.savez(os.path.join(output_path, 'calibration.npz'), mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
 print('finished calibration')

 s = time.time()
 dst = cv.undistort(last_image, mtx, dist, None)
 e = time.time()
 print(f'undistort time: {e-s}')
 plot_2_imgs(last_image, dst)


def undistort_video(video_path, calibration_path, start_frame=0):
 undist = undistort(calibration_path)

 rotate = 2 if 'ZED' in video_path else 1
 cam = video_wrapper(video_path, rotate)

 fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))

 id_ = 0
 ret = True
 while ret:
  id_ += 1
  print(id_)
  ret, img = cam.get_frame()

  if id_ < start_frame:
   id_ += 1
   continue

  if ret:
   undist_image = undist.apply(img)

   smaller_size_50 = (int(img.shape[1] * 0.5), int(img.shape[0] * 0.5))
   smaller_img = cv.resize(img, smaller_size_50)
   smaller_undist_img = cv.resize(undist_image, smaller_size_50)
   full_image = np.hstack([smaller_img, smaller_undist_img])
   cv.imshow('compare', full_image)
   cv.waitKey(100)

 cv.destroyAllWindows()



if __name__ == "__main__":
 video_path = "/media/matans/My Book/FruitSpec/target/wetransfer_target_2023-10-24_1153/target/row_31/1/Result_RGB.mkv"
 output_path = "/media/matans/My Book/FruitSpec/target/jai_83"
 #validate_output_path(output_path)
 #calibrate_lense(video_path, output_path, start_frame=170, view=True)

 calibration_path = '/media/matans/My Book/FruitSpec/target/calibration.npz'
 video_path = '/media/matans/My Book/FruitSpec/target/test/Result_FSI_1.mkv'
 undistort_video(video_path, calibration_path)