import os
import csv
import cv2

from vision.vizualization.drawer import draw_rectangle, draw_text, draw_highlighted_test, get_color


class ResultsCollector():

    def __init__(self):

        self.detections = []
        self.tracks = []
        self.file_names = []
        self.file_ids = []



    def collect_detections(self, detection_results, img_id, scale_):
        if detection_results is not None:
            output = []
            for det in detection_results:
                det.append(img_id)
                output.append(det)
            self.detections += output


    @staticmethod
    def map_det_2_trck(t2d_mapping, number_of_detections):

        if t2d_mapping is not None:
            track_ids = []

            d2t = {v: k for k, v in sorted(t2d_mapping.items(), key=lambda item: item[1])}
            det_ids = list(d2t.keys())
            for i in range(number_of_detections):
                if i in det_ids:
                    track_ids.append(d2t[i])
                else:
                    track_ids.append(-1)
        else:
            track_ids = [-1 for _ in range(number_of_detections)]

        return track_ids


    def collect_tracks(self, tracking_results):

        #output_len = len(tracking_results[2])

        #frame_ids = [tracking_results[0] for _ in range(output_len)]
        #bboxes = tracking_results[1]
        #tracker_ids = tracking_results[2]
        #tracker_score = tracking_results[3]

        #output = list(map(self.single_tracking_to_list, frame_ids, tracker_ids, tracker_score, bboxes))
        self.tracks += tracking_results

        return tracking_results

    def collect_file_name(self, file_anme):
        self.file_names.append(file_anme)

    def collect_id(self, id_):
        self.file_ids.append(id_)


    @staticmethod
    def single_detection_to_list(detection, image_id, scale_):
        # Detection ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        x1 = int(detection[0] * scale_)
        y1 = int(detection[1] * scale_)
        x2 = int(detection[2] * scale_)
        y2 = int(detection[3] * scale_)
        obj_conf = detection[4]
        class_conf = detection[5]
        class_pred = detection[6]

        # res ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred, image_id)
        res = [x1, y1, x2, y2, obj_conf, class_conf, class_pred, image_id]

        return res

    @staticmethod
    def single_tracking_to_list(frame_id, track_id, tracker_score, bbox):
        # bbox: [x1, y1, w, h]

        x1 = int(bbox[0])
        y1 = int(bbox[1])
        x2 = x1 + int(bbox[2])
        y2 = y1 + int(bbox[3])
        return [x1, y1, x2, y2, tracker_score, track_id, frame_id]


    def dump_to_csv(self, output_file_path, detections=True):

        if detections:
            fields = ["x1", "y1", "x2", "y2", "obj_conf", "class_conf",
                      "image_id", "class_pred"]
            rows = self.detections
        else:
            fields = ["bbox", "tracker_score", "frame_ids", "track_id"]
            rows = self.tracks

        with open(output_file_path, 'w') as f:
            # using csv.writer method from CSV package
            write = csv.writer(f)
            write.writerow(fields)
            write.writerows(rows)
        print(f'Done writing results to csv')


    def write_results_on_movie(self, movie_path, output_path, write_tracks=True, write_frames=False):
        """
            the function draw results on each frame
            - Each frame will be saved with results if 'write_frames' is True
            - 'write_tracks' indicate if the tracker results will be drawn or detections
        """
        if write_tracks:
            hash = self.create_hash(self.tracks)
        else:
            hash = self.create_hash(self.detections)

        cap = cv2.VideoCapture(movie_path)

        if not write_frames:
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = cap.get(cv2.CAP_PROP_FPS)
            output_video_name = os.path.join(output_path, 'result_video.mp4')
            output_video = cv2.VideoWriter(output_video_name, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                     fps, (width, height))
        # Read until video is completed
        f_id = 0
        while (cap.isOpened()):

            ret, frame = cap.read()
            if ret == True:
                dets = hash[f_id]
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if write_frames:
                    self.draw_and_save(frame, dets, f_id, output_path)
                else:
                    frame = self.draw_dets(frame, dets)
                    output_video.write(frame)

                f_id += 1

        if not write_frames:
            output_video.release()
        cap.release()
        cv2.destroyAllWindows()

    def draw_and_save_dir(self, data_dir, output_path, tracks=False):
        if tracks:
            hash_ = self.create_hash(self.tracks)
        else:
            hash_ = self.create_hash(self.detections)
        img_with_dets = list(hash_.keys())
        for i, id_ in enumerate(self.file_ids):
            if id_ in img_with_dets:
                dets = hash_[id_]
            else:
                dets = []

            frame = cv2.imread(os.path.join(data_dir, self.file_names[i]))
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            self.draw_and_save(frame, dets, id_, output_path)

    def draw_and_save(self, frame, dets, f_id, output_path):

       frame = self.draw_dets(frame, dets)
       output_file_name = os.path.join(output_path, f'frame_{f_id}_res.jpg')
       frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
       cv2.imwrite(output_file_name, frame)


    def draw_dets(self,frame, dets):

        for det in dets:
            track_id = det[-2]
            color_id = int(track_id) % 15 # 15 is the number of colors in list
            color = get_color(color_id)
            text_color = get_color(-1)
            frame = draw_rectangle(frame, (int(det[0]), int(det[1])), (int(det[2]), int(det[3])), color, 3)
            frame = draw_highlighted_test(frame, f'ID:{track_id}', (det[0], det[1]), frame.shape[1], color, text_color,
                                          True, 10, 3)

        return frame

    @staticmethod
    def create_hash(detections):

        hash = {}
        id_list = []
        for det in detections:
            if det[-1] in id_list: # -1 is image id
                hash[det[-1]].append(det)
            else:
                hash[det[-1]] = [det]
                id_list.append(det[-1])

        return hash


def scale(det_dims, frame_dims):

    r = min(det_dims[0] / frame_dims[0], det_dims[1] / frame_dims[1])
    return (1 / r)


def scale_det(detection, scale_):
    # Detection ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
    # Detection ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
    x1 = int(detection[0] * scale_)
    y1 = int(detection[1] * scale_)
    x2 = int(detection[2] * scale_)
    y2 = int(detection[3] * scale_)
    obj_conf = detection[4]
    class_conf = detection[5]
    class_pred = detection[6]

    # res ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred, image_id)
    return [x1, y1, x2, y2, obj_conf, class_conf, class_pred]
