



class ResultsManager():

    def __init__(self):

        self.detections = []
        self.tracks = []

    def collect_detections(self, detection_results, img_id):

        if detection_results is not None:
            det = detection_results[0].to('cpu').numpy()

            temp_id_list = [img_id for _ in range(det.shape[0])]
            self.detections += list(map(self.single_detection_to_list, det, temp_id_list))


    def collect_tracks(self, tracking_results):

        output_len = len(tracking_results[2])

        frame_ids = [tracking_results[0] for _ in range(output_len)]
        bboxes = tracking_results[1]
        tracker_ids = tracking_results[2]
        tracker_score = tracking_results[3]

        self.tracks += list(map(self.single_tracking_to_list, frame_ids, tracker_ids, tracker_score, bboxes))


    @staticmethod
    def single_detection_to_list(detection, image_id):

        res = list(detection)
        res.append(image_id)

        return res

    @staticmethod
    def single_tracking_to_list(frame_ids, track_id, tracker_score, bbox):

        return [frame_ids, track_id, tracker_score, bbox]

