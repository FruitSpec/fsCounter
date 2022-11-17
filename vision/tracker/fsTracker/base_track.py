from collections import deque


class TrackState(object):
    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class Track:

    def __init__(self):
        self._count = 0
        self.track_id = 0
        self.is_activated = False
        self.state = None
        self.bbox = None  # x1y1x2y2
        self.score = None
        self.cls = None
        self.frame_size = []
        self.accumulated_dist = deque()


    def get_track_search_window(self, search_window, margin=15, multiply=1.25):
        tx = search_window[0] * multiply
        ty = search_window[1] * multiply

        # positive tx - objects are moving to the left
        if tx > 0:
            # positive ty - objects are moving up
            if ty > 0:
                x1 = max(self.bbox[0] - tx, 0)  # bbox[0] is x1
                x2 = min(self.bbox[2] + margin, self.frame_size[1])  # bbox[2] is x2
                y1 = max(self.bbox[1] - ty, 0)  # bbox[1] is y1
                y2 = min(self.bbox[3] + margin, self.frame_size[0])  # bbox[3] is y2
            # negative ty - objects are moving down
            else:
                x1 = max(self.bbox[0] - tx, 0)  # bbox[0] is x1
                x2 = min(self.bbox[2] + margin, self.frame_size[1])  # bbox[2] is x2
                y1 = max(self.bbox[1] - margin, 0)  # bbox[1] is y1
                y2 = min(self.bbox[3] - ty, self.frame_size[0])  # bbox[3] is y2
        # negative tx - objects are moving right
        else:
            # positive ty - objects are moving up
            if ty > 0:
                x1 = max(self.bbox[0] - margin, 0)  # bbox[0] is x1
                x2 = min(self.bbox[2] - tx, self.frame_size[1])  # bbox[2] is x2
                y1 = max(self.bbox[1] - ty, 0)  # bbox[1] is y1
                y2 = min(self.bbox[3] + margin, self.frame_size[0])  # bbox[3] is y2
            # negative ty - objects are moving down
            else:
                x1 = max(self.bbox[0] - margin, 0)  # bbox[0] is x1
                x2 = min(self.bbox[2] - tx, self.frame_size[1])  # bbox[2] is x2
                y1 = max(self.bbox[1] - margin, 0)  # bbox[1] is y1
                y2 = min(self.bbox[3] - ty, self.frame_size[0])  # bbox[3] is y2


        return x1, y1, x2, y2

    def add(self, det, id_, frame_size):
        self.track_id = id_
        self.bbox = [det[0], det[1], det[2], det[3]]
        self.is_activated = True
        self.score = det[4] * det[5]
        self.cls = det[6]
        self._count += 1
        self.frame_size = frame_size
        self.state = TrackState.New



    def update(self, det):
        bbox = [det[0], det[1], det[2], det[3]]
        self.accumulated_dist.append(self.bbox[0] - bbox[0])
        if self.accumulated_dist.__len__() > 3:
            self.accumulated_dist.popleft()
        self.bbox = bbox
        self.is_activated = True
        self.score = det[4] * det[5]
        self.cls = det[6]
        self._count += 1
        self.state = TrackState.Tracked

    def output(self):
        return [self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3], self.score, self.cls, self.track_id]