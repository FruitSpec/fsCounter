import csv


class ResultsCollector():

    def __init__(self):

        self.detections = []
        self.tracks = []


    def collect_detections(self, detection_results, t2d_mapping, img_id):
        if detection_results is not None:
            det = detection_results[0].to('cpu').numpy()
            n_detections = det.shape[0]
            track_ids = self.map_det_2_trck(t2d_mapping, n_detections)
            temp_id_list = [img_id for _ in range(n_detections)]

            output = list(map(self.single_detection_to_list, det, temp_id_list, track_ids))
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


    def collect_tracks(self, tracking_results, t2d_mapping):

        output_len = len(tracking_results[2])
        t2d_list = [[k, v] for k, v in t2d_mapping.items()]

        frame_ids = [tracking_results[0] for _ in range(output_len)]
        bboxes = tracking_results[1]
        tracker_ids = tracking_results[2]
        tracker_score = tracking_results[3]

        output = list(map(self.single_tracking_to_list, frame_ids, tracker_ids, tracker_score, bboxes, t2d_list))
        self.tracks += output

        return output



    @staticmethod
    def single_detection_to_list(detection, image_id, track_id):
        # Detection ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        res = list(detection)
        res.append(image_id)
        res.append(track_id)

        # res ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred, image_id, track_id)
        return res

    @staticmethod
    def single_tracking_to_list(frame_ids, track_id, tracker_score, bbox, t2d):

        if t2d[0] != track_id:
            print('Results_Manager: tracker id and detection id mismatch')
            det_id = None
        else:
            det_id = t2d[1]

        return [bbox, tracker_score, det_id, frame_ids, track_id]


    def dump_to_csv(self, output_file_path, detections=True):

        if detections:
            fields = ["x1", "y1", "x2", "y2", "obj_conf", "class_conf",
                      "class_pred", "image_id", "track_id"]
            rows = self.detections
        else:
            fields = ["bbox", "tracker_score", "det_id", "frame_ids", "track_id"]
            rows = self.tracks

        with open(output_file_path, 'w') as f:
            # using csv.writer method from CSV package
            write = csv.writer(f)
            write.writerow(fields)
            write.writerows(rows)
        print(f'Done writing results to csv')
