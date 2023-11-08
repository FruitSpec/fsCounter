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


 def apply_on_batch(self, batch):
  output = []
  for img in batch:
   output.append(cv.undistort(img, self.mtx, self.dist, None))

  return output





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
 treshold = 50 if 'ZED' in video_path else 20
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
   chees_img[chees_img > treshold] = 255
   chees_img[chees_img <= treshold] = 0

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

 id_ = 0
 ret = True
 while ret:
  id_ += 1
  print(id_)
  if rotate == 2:
   cam.grab()
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

def get_cameras_frame(r_id, z_id, cam_z=None, cam_r=None, zed_video=None, rgb_video=None, display=False, single_image_size=(768, 1024)):
 if cam_z is None and zed_video is not None:
  rotate_z = 2 if 'ZED' in zed_video else 1
  cam_z = video_wrapper(zed_video, rotate_z)
 else:
  print(f'NO ZED CAM')
  return None, None

 if cam_r is None and rgb_video is not None:
  rotate_r = 2 if 'ZED' in rgb_video else 1
  cam_r = video_wrapper(rgb_video, rotate_r)
 else:
  print(f'NO JAI CAM')
  return None, None

 f_count = 0
 while f_count < z_id:
  cam_z.grab()
  ret_z, frame_z = cam_z.get_frame()

  if not ret_z:
   print(f'FAILED TO LOAD FRAME {f_id}')
   return
  f_count += 1

 f_count = 0
 while f_count < r_id:
  ret_r, frame_r = cam_r.get_frame()

  if not ret_r:
   print(f'FAILED TO LOAD FRAME {f_id}')
   return
  f_count += 1


 if display:
  display_side_by_side(frame_z, frame_r, single_image_size)

 return frame_z, frame_r

def display_side_by_side(left, right, single_image_size=(768, 1024)):
 disp_l = cv.resize(left, single_image_size)
 disp_r = cv.resize(right, single_image_size)

 canvas = np.hstack([disp_l, disp_r])
 plt.figure()
 plt.imshow(canvas)
 plt.title(f'Side by side')
 plt.show()

def binary_clip(img, threshold):
 b_img = img.copy()
 b_img[b_img > threshold] = 255
 b_img[b_img <= threshold] = 0

 return b_img


def save_to_frames(zed_video, rgb_video, output_path):
 rotate_z = 2 if 'ZED' in zed_video else 1
 cam_z = video_wrapper(zed_video, rotate_z)

 rotate_r = 2 if 'ZED' in rgb_video else 1
 cam_r = video_wrapper(rgb_video, rotate_r)

 ret_z = True
 f_count = 0
 while ret_z:
  cam_z.grab()
  ret_z, frame_z = cam_z.get_frame()
  if ret_z:
   cv.imwrite(os.path.join(output_path, f'zed_f{f_count}.jpg'), frame_z)
   f_count += 1

 ret_r = True
 f_count = 0
 while ret_r:
  ret_r, frame_r = cam_r.get_frame()
  if ret_r:
   cv.imwrite(os.path.join(output_path, f'jai_f{f_count}.jpg'), frame_r)
   f_count += 1

def get_cameras_ratio_and_location(corners_z, corners_r, z_size=(1920, 1080), r_size=(2048, 1536), pattern_size=(6,9)):
 z_x_corr = corners_z[:, 0, 0].reshape(pattern_size[1], pattern_size[0])
 z_y_corr = corners_z[:, 0, 1].reshape(pattern_size[1], pattern_size[0])

 r_x_corr = corners_r[:, 0, 0].reshape(pattern_size[1], pattern_size[0])
 r_y_corr = corners_r[:, 0, 1].reshape(pattern_size[1], pattern_size[0])

 delta_x_z = np.mean(z_x_corr[:, -1] - z_x_corr[:, 0])
 delta_x_r = np.mean(r_x_corr[:, -1] - r_x_corr[:, 0])

 z_to_r_x_ratio = delta_x_z / delta_x_r

 delta_y_z = np.mean(z_y_corr[-1, :] - z_y_corr[0, :])
 delta_y_r = np.mean(r_y_corr[-1, :] - r_y_corr[0, :])

 z_to_r_y_ratio = delta_y_z / delta_y_r

 z_y_roi_in_r = z_size[0] / z_to_r_y_ratio
 z_x_roi_in_r = z_size[1] / z_to_r_x_ratio

 z_x_corr_in_r_size = z_x_corr / z_to_r_x_ratio
 z_y_corr_in_r_size = z_y_corr / z_to_r_y_ratio

 tx = np.mean(z_x_corr_in_r_size - r_x_corr)
 ty = np.mean(z_y_corr_in_r_size - r_y_corr)

 return z_to_r_x_ratio, z_to_r_y_ratio, z_x_roi_in_r, z_y_roi_in_r, tx, ty

def crop_black(image):
 non_black_coords = np.column_stack(np.where(image[:, :, 0] > 0))

 # Get the cropping coordinates
 top, left = non_black_coords.min(axis=0)
 bottom, right = non_black_coords.max(axis=0) + 1

 return image[top:bottom, left:right, :]
if __name__ == "__main__":
 zed_video = "/media/matans/My Book/FruitSpec/target/wetransfer_target_2023-10-24_1153/target/row_31/1/ZED.mkv"
 rgb_video = "/media/matans/My Book/FruitSpec/target/wetransfer_target_2023-10-24_1153/target/row_31/1/Result_RGB.mkv"
 output_path = "/media/matans/My Book/FruitSpec/target/row_31_frames"
 validate_output_path(output_path)
 #calibrate_lense(video_path, output_path, start_frame=170, view=True)


 calibration_path_r = '/media/matans/My Book/FruitSpec/target/calibration.npz'
 calibration_path_z = '/media/matans/My Book/FruitSpec/target/zed/calibration.npz'
 # video_path = '/media/matans/My Book/FruitSpec/target/test/Result_FSI_1.mkv'
 #video_path = '/media/matans/My Book/FruitSpec/target/test/ZED_1.svo'
 #undistort_video(zed_video,calibration_path_z,0)
 f_id = 250
 undis_r = undistort(calibration_path_r)
 undis_z = undistort(calibration_path_z)
 frame_z, frame_r = get_cameras_frame(176, 206, zed_video=zed_video, rgb_video=rgb_video, single_image_size=(768, 1024))

 undis_r = undis_r.apply(frame_r)
 undis_z = undis_z.apply(frame_z)
 display_side_by_side(frame_z, frame_r)
 display_side_by_side(frame_z, undis_r)
 """ Verify"""
 undis_z = frame_z
 #undis_z = crop_black(undis_z)

 binary_z = binary_clip(undis_z[:,:,0], 50)
 binary_r = binary_clip(undis_r[:,:,0], 30)

 _, _, corners_z, ret_z = get_corners([], [], binary_z, (6, 9), [], resize=True)
 _, _, corners_r, ret_r = get_corners([], [], binary_r, (6, 9), [], resize=True)


 cv.drawChessboardCorners(undis_z, (6, 9), corners_z, ret_z)
 cv.drawChessboardCorners(undis_r, (6, 9), corners_r, ret_r)

 #display_side_by_side(undis_z, undis_r)

 z_to_r_x_ratio, z_to_r_y_ratio, z_x_roi_in_r, z_y_roi_in_r, tx, ty = get_cameras_ratio_and_location(corners_z, corners_r, z_size=(undis_z.shape[0], undis_z.shape[1]))

 undis_r_crop = undis_r[int(-ty):int(-ty + z_y_roi_in_r), int(-tx): int(-tx + z_x_roi_in_r), :]

 display_side_by_side(undis_z, undis_r_crop)

 print(f'z to r x ratio: {z_to_r_x_ratio}')
 print(f'z to r y ratio: {z_to_r_y_ratio}')
 print(f'z x roi: {z_x_roi_in_r}')
 print(f'z y roi: {z_y_roi_in_r}')

 print('Done')
 #undistort_video(video_path, calibration_path)