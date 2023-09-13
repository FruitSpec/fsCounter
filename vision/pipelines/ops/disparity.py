import cv2
from omegaconf import OmegaConf

from vision.kp_matching.infer import lightglue_infer
from vision.kp_matching.sp_lg.utils import numpy_image_to_torch
from vision.tools.image_stitching import resize_img
from vision.tools.video_wrapper import video_wrapper
from vision.misc.help_func import get_repo_dir, validate_output_path

class disparity():
    def __init__(self, cfg):

        self.lg = lightglue_infer(cfg)
    def calc_disparity(self, imgL, imgR):
        stereo = cv2.StereoBM_create(numDisparities=64, blockSize=7)
        disparity = stereo.compute(imgL, imgR)

        return disparity


    def get_disparity(self, imgL, imgR):

        prepL, r = self.preprocess(imgL)
        prepR, _ = self.preprocess(imgR)
        points0, points1, matches = self.lg.match(prepL, prepR)

        points0 = points0.cpu().numpy()
        points1 = points1.cpu().numpy()

        M, st = self.lg.calcaffine(points0, points1)
        tx = int(M[0, 2] / r)
        ty = int(M[1, 2] / r )


        if ty < 0:
            inputL = imgL[(-1) * ty:imgL.shape[0], (-1) * tx: imgL.shape[1], :]
            inputR = imgR[:imgR.shape[0] + ty, :imgR.shape[1]+ tx, :]

        else:
            inputL = imgL[:imgL.shape[0] - ty, :, :]
            inputR = imgR[ty:imgR.shape[0], :, :]

        inputL = cv2.cvtColor(inputL, cv2.COLOR_BGR2GRAY)
        inputR = cv2.cvtColor(inputR, cv2.COLOR_BGR2GRAY)
        disp = self.calc_disparity(inputL, inputR)




    def preprocess(self, img, downscale=4, to_tensor=True):

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        input_, r = resize_img(img_gray, img_gray.shape[0] // downscale)

        if to_tensor:
            input_ = numpy_image_to_torch(input_).cuda()

        return input_, r



if __name__ == "__main__":
    repo_dir = get_repo_dir()
    pipeline_config = "/vision/pipelines/config/pipeline_config.yaml"
    cfg = OmegaConf.load(repo_dir + pipeline_config)

    video_path = "/media/matans/My Book/FruitSpec/Mehadrin/00608060/210823/row_3/1/Result_FSI.mkv"
    f_id = 55

    cam = video_wrapper(video_path, rotate=1)
    d = disparity(cfg)

    ret_r, img_right = cam.get_frame(f_id)
    ret_l, img_left = cam.get_frame(f_id + 1)

    d.get_disparity(img_left, img_right)

