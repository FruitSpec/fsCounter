



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

        frame_id = tracking_results[0]
        bboxes = tracking_results[1]
        tracker_ids = tracking_results[2]
        detection_score = tracking_results[3]




    @staticmethod
    def single_detection_to_list(detection, image_id):

        res = list(detection)
        res.append(image_id)

        return res