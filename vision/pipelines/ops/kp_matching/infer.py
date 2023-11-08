import os
import cv2
import numpy as np


from vision.pipelines.ops.kp_matching.sp_lg import LightGlue, SuperPoint, DISK
from vision.pipelines.ops.kp_matching.sp_lg.utils import rbd, numpy_image_to_torch
from vision.tools.image_stitching import resize_img


class lightglue_infer():

    def __init__(self, cfg, type='superpoint', len_size=61):
        """
        type can be 'superpoint' or 'disk'
        """
        if type == 'superpoint':
            self.extractor = SuperPoint(max_num_keypoints=512).eval().cuda()  # load the extractor
        elif type == 'disk':
            self.extractor = DISK(max_num_keypoints=512).eval().cuda()  # load the extractor

        self.matcher = LightGlue(features=type, depth_confidence=0.9, width_confidence=0.95).eval().cuda()  # load the matcher
        self.zed_roi_params = cfg.sensor_aligner.zed_roi_params if len_size == 61 else cfg.sensor_aligner.zed_roi_params_83
        self.y_s, self.y_e, self.x_s, self.x_e = self.zed_roi_params.values()
        self.size = cfg.sensor_aligner.size
        self.sx = cfg.sensor_aligner.sx if len_size == 61 else cfg.sensor_aligner.sx_83
        self.sy = cfg.sensor_aligner.sy if len_size == 61 else cfg.sensor_aligner.sy_83
        self.roix = cfg.sensor_aligner.roix if len_size == 61 else cfg.sensor_aligner.roix_83
        self.roiy = cfg.sensor_aligner.roiy if len_size == 61 else cfg.sensor_aligner.roiy_83
        self.len_size = len_size
        self.zed_size = [1920, 1080]
        self.jai_size = [2048, 1536]
        self.batch_size = cfg.batch_size
        self.last_feat = None

    def to_tensor(self, image):

        return numpy_image_to_torch(image).cuda()


    def match(self, input0, input1):
        feats0 = self.extractor.extract(input0)
        feats1 = self.extractor.extract(input1)

        # match the features
        matches01 = self.matcher({'image0': feats0, 'image1': feats1})
        feats0, feats1, matches01 = [rbd(x) for x in [feats0, feats1, matches01]]  # remove batch dimension
        matches = matches01['matches']  # indices with shape (K,2)
        points0 = feats0['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
        points1 = feats1['keypoints'][matches[..., 1]]

        return points0, points1, matches

    def preprocess_images(self, zed, jai_rgb, downscale=1, to_tensor=True):

        if self.len_size != 83:
            cropped_zed = zed[self.y_s: self.y_e, self.x_s:self.x_e, :]
        else:
            cropped_zed = zed
        #input_zed = cv2.resize(cropped_zed, int(cropped_zed.shape[1] / self.sx), int(cropped_zed.shape[0] / self.sy)))
        input_zed = zed


        input_zed, rz = resize_img(input_zed, input_zed.shape[0] // downscale)
        input_jai, rj = resize_img(jai_rgb, jai_rgb.shape[0] // downscale)

        if to_tensor:
            input_zed = self.to_tensor(input_zed)
            input_jai = self.to_tensor(input_jai)

        return input_zed, rz, input_jai, rj


    @staticmethod
    def calcaffine(src_pts, dst_pts):
        if dst_pts.__len__() > 0 and src_pts.__len__() > 0:  # not empty - there was a match
            M, status = cv2.estimateAffine2D(src_pts, dst_pts)
        else:
            M = None
            status = []

        return M, status

    def get_tx_ty(self, M, st, rz):
        # in case no matches or less than 5 matches
        if len(st) == 0 or np.sum(st) <= 5:
            print('failed to align, using center default')
            tx = -999
            ty = -999
            # roi in frame center
            mid_x = (self.x_s + self.x_e) // 2
            mid_y = (self.y_s + self.y_e) // 2
            x1 = mid_x - (self.roix // 2)
            x2 = mid_y + (self.roix // 2)
            y1 = mid_y - (self.roiy // 2)
            y2 = mid_y + (self.roiy // 2)
        else:



            if self.len_size != 83:
                tx = M[0, 2] * (-1)
                ty = M[1, 2] * (-1)
                tx = tx / rz * self.sx
                ty = ty / rz * self.sy
                # tx = np.mean(deltas[:, 0, 0]) / rz * sx
                # ty = np.mean(deltas[:, 0, 1]) / rz * sy

                x1, y1, x2, y2 = self.get_zed_roi(tx, ty)
            else:
                tx = M[0, 2] / rz * (-1)
                ty = M[1, 2] / rz * (-1)
                sx = M[0, 0]
                sy = M[1, 1]
                x1, y1, x2, y2 = self.get_jai_roi(tx, ty, sx ,sy)
        return (x1, y1, x2, y2), tx, ty, int(np.sum(st))

    def get_zed_roi(self, tx, ty):

        if tx < 0:
            x1 = 0
            x2 = self.roix
        elif tx + self.roix > self.zed_size[1]:
            x2 = self.zed_size[1]
            x1 = self.zed_size[1] - self.roix
        else:
            x1 = tx
            x2 = tx + self.roix

        if ty < 0:
            y1 = self.y_s + ty
            if y1 < 0:
                y1 = self.y_s
            y2 = y1 + self.roiy
        elif ty + self.roiy > self.zed_size[0]:
            y2 = self.y_e
            y1 = self.y_e - self.roiy
        else:
            y1 = self.y_s + ty
            y2 = self.y_s + ty + self.roiy

        return x1, y1, x2, y2

    def get_jai_roi(self, tx, ty, sx, sy):

        roix = self.zed_size[1] * sx
        full_roiy = self.zed_size[0] * sy

        roiy = min(self.jai_size[0], full_roiy)
        #x1 = (self.jai_size[1] // 2) + tx - (self.roix // 2)
        x1 = tx

        if x1 < 0:
            x1 = 0
            #x2 = self.roix
            x2 = roix
        else:
            #x2 = x1 + self.roix
            x2 = x1 + roix
            if x2 > self.jai_size[1]:
                x2 = self.jai_size[1]
                #x1 = x2 - self.roix
                x1 = x2 - roix

        #y1 = (self.jai_size[0] // 2) + ty - (self.roiy // 2)
        y1 = (self.jai_size[0] // 2) + ty - (full_roiy // 2)

        if y1 < 0:
            y1 = 0
            #y2 = self.roiy
            y2 = roiy
        else:
            #y2 = y1 + self.roiy
            y2 = y1 + roiy
            if y2 > self.jai_size[0]:
                y2 = self.jai_size[0]
                #y1 = y2 - self.roiy
                y1 = y2 - roiy


        #if tx < 0:
        #    x1 = (self.jai_size[1] // 2) + tx - (self.roix // 2)
        #    x2 = x1 + self.roix
        #elif tx + self.roix > self.jai_size[1]:
        #    x2 = self.jai_size[1]
        #    x1 = self.jai_size[1] - self.roix
        #else:
        #    x1 = (self.jai_size[1] // 2) + tx
        #    x2 = tx + self.roix

        #if ty < 0:
        #    y1 = 0
        #    y2 = y1 + self.roiy
        #elif ty + self.roiy > self.jai_size[0]:
        #    y2 = self.jai_size[0]
        #    y1 = y2 - self.roiy
        #else:
        #    y1 = ty
        #    y2 = ty + self.roiy

        return x1, y1, x2, y2



    def align_sensors(self, zed, jai_rgb, debug=None, method = "affine"):

        zed_input, rz, jai_input, rj = self.preprocess_images(zed, jai_rgb)

        #zed_input = zed_input[:,50:1700,:]

        points0, points1, matches = self.match(zed_input, jai_input)

        points0 = points0.cpu().numpy()
        points1 = points1.cpu().numpy()

        if self.len_size != 83:
            M, st = self.calcaffine(points0, points1)
        else:
            M, st = self.calcaffine(points1, points0)

        if debug is not None:
            zed_debug, _, jai_debug, _ = self.preprocess_images(zed, jai_rgb, to_tensor=False)
            out_img = draw_matches(zed_debug, jai_debug, points0, points1)
            cv2.imwrite(os.path.join(debug[0]['output_path'], f"alignment_f{debug[0]['f_id']}.jpg"),
                        out_img)
        if method == "delta":
            deltas = np.array(points0) - np.array(points1)
            tx = np.mean(deltas[:, 0]) / rz * self.sx
            ty = np.mean(deltas[:, 1]) / rz * self.sy
            x1, y1, x2, y2 = self.get_zed_roi(tx, ty)
            return (x1, y1, x2, y2), tx, ty, int(np.sum(st))

        if self.len_size != 83:

            return self.get_tx_ty(M, st, rz)

        else:
            rgb_output = cv2.warpAffine(jai_rgb, M, [self.zed_size[1], self.zed_size[0]])
            #valid_rgb = remove_black_area(rgb_output)
            h, w, c = jai_rgb.shape
            corners = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], dtype=np.float32)
            transformed_corners = np.dot(M, corners.T).T

            h, w, _ = rgb_output.shape
            corners1 = np.array([[0, 0, 1], [w, 0, 1], [0, h, 1], [w, h, 1]], dtype=np.float32)
            transformed_corners1 = np.dot(M, corners1.T).T

            print(transformed_corners)

    def align_on_batch(self, zed_batch, jai_batch, workers=4, debug=None):
        if False:#len(zed_batch) < 1:
            corr, tx, ty = align_sensors_cuda(zed_batch[0], jai_batch[0])
            results = [[corr, tx, ty]]
        else:
            zed_input = []
            jai_input = []
            streams = []
            for z, j in zip(zed_batch, jai_batch):
                if z is not None and j is not None:
                    zed_input.append(z)
                    jai_input.append(j)
                    #streams.append(cv2.cuda_Stream())
            if debug is None:
                debug = [None] * self.batch_size
            results = list(map(self.align_sensors, zed_input, jai_input, debug))


        output = []
        for r in results:
            output.append([r[0], r[1], r[2], r[3]])

        return output


    def batch_translation(self, batch):

        inputs, rs = self.preprocess_batch(batch)
        feats_batch = list(map(self.extractor.extract, inputs))

        matcher_output = self.batch_matcher(feats_batch)
        txs, tys = self.get_tx_ty_from_outputs(matcher_output, rs)

        res = self.pack_results(txs, tys, len(batch))

        return res

    @staticmethod
    def pack_results(txs, tys, batch_size):
        if len(txs) < batch_size: # last_feats is None
            txs = [None] + txs
            tys = [None] + tys

        res = []
        for tx, ty in zip(txs, tys):
            res.append((tx, ty))

        return res

    def preprocess_batch(self, batch, downscale=4, to_tensor=True):

        downscale_list = [img.shape[0] // downscale for img in batch]
        results = list(map(resize_img, batch, downscale_list))

        if to_tensor:
            imgs = [result[0] for result in results]
            output = list(map(self.to_tensor, imgs))
        else:
            output = [result[0] for result in results]

        rs = [result[1] for result in results]

        return output, rs


    def batch_matcher(self, feats_batch):

        inputs = self.batch_to_feats_inputs(feats_batch)
        if inputs:
            outputs = list(map(self.execute_matcher, inputs))
        else:
            outputs = []

        self.last_feat = feats_batch[-1]

        return outputs


    def get_tx_ty_from_outputs(self, outputs, rs):

        txs = []
        tys = []
        if outputs:
            for output, r in zip(outputs, rs):
                M, st = self.calcaffine(output[0], output[1])

                if len(st) == 0 or np.sum(st) <= 5:
                    print('failed to align, using center default')
                    txs.append(-999)
                    tys.append(-999)
                else:

                    txs.append((-1) * M[0, 2] / r)
                    tys.append((-1) * M[1, 2] / r)
                    #txs.append(M[0, 2] / r)
                    #tys.append(M[1, 2] / r)

        return txs, tys





    def execute_matcher(self, input_):
        matches01 = self.matcher(input_)
        feats0, feats1, matches01 = [rbd(x) for x in [input_['image0'], input_['image1'], matches01]]  # remove batch dimension
        matches = matches01['matches']  # indices with shape (K,2)
        points0 = feats0['keypoints'][matches[..., 0]]  # coordinates in image #0, shape (K,2)
        points1 = feats1['keypoints'][matches[..., 1]]

        points0 = points0.cpu().numpy()
        points1 = points1.cpu().numpy()

        return points0, points1, matches



    def batch_to_feats_inputs(self, feats_batch):

        input_ = []
        if self.last_feat is not None:
            input_.append({'image0': self.last_feat, 'image1': feats_batch[0]})
        for i in range(1, len(feats_batch)):
            input_.append({'image0': feats_batch[i-1], 'image1': feats_batch[i]})

        return input_
def inference(extractor, image, batch_queue):
    kp = extractor.extract(image)
    batch_queue.put(kp)
    return batch_queue





def draw_matches(input0, input1, points0, points1):

    h = max(input0.shape[0], input1.shape[0])
    w = input0.shape[1] + input1.shape[1] + 1
    canvas = np.zeros((h, w, 3))

    canvas[:input0.shape[0], :input0.shape[1], :] = input0
    canvas[:input1.shape[0], input0.shape[1]: input0.shape[1] + input1.shape[1], :] = input1

    for p0, p1 in zip(points0, points1):
        canvas = cv2.circle(canvas, (int(p0[0]), int(p0[1])), 3, (255, 0, 0))
        canvas = cv2.circle(canvas, (input0.shape[1] + int(p1[0]), int(p1[1])), 3, (255, 0, 0))
        canvas = cv2.line(canvas, (int(p0[0]), int(p0[1])), (input0.shape[1] + int(p1[0]), int(p1[1])), (0, 255, 0))

    return canvas

