import numpy as np
from collections import deque
# from numba import jitclass, types


# track_spec = [
#     ('_count', types.int32),
#     ('track_id', types.int32),
#     ('is_activated', types.boolean),
#     ('bbox', types.float32[:]),
#     ('score', types.float32),
#     ('cls', types.int32),
#     ('frame_size', types.int32[:]),
#     ('accumulated_dist', types.ListType(types.float32)),
#     ('accumulated_height', types.ListType(types.float32)),
#     ('lost_counter', types.int32),
#     ('depth', types.float32)]

#track_class_type = types.Record.make_c_struct(track_spec)

# Decorate the class with jitclass using the spec
#@jitclass(track_spec)
class Track:

    def __init__(self):
        self._count = 0
        self.track_id = 0
        self.is_activated = False
        #self.state = None
        self.bbox = None  # x1y1x2y2
        self.score = None
        self.cls = None
        self.frame_size = []
        self.accumulated_dist = []
        self.accumulated_height = []
        self.lost_counter = 0
        self.depth = np.nan


    # def get_track_search_window(self, search_window, margin=15, multiply=1.25):
    #     tx = search_window[0] * multiply
    #     ty = search_window[1] * multiply
    #
    #     # positive tx - objects are moving to the left
    #     if tx > 0:
    #         # positive ty - objects are moving up
    #         if ty > 0:
    #             x1 = max(self.bbox[0] - tx, 0)  # bbox[0] is x1
    #             x2 = min(self.bbox[2] + margin, self.frame_size[1])  # bbox[2] is x2
    #             y1 = max(self.bbox[1] - ty, 0)  # bbox[1] is y1
    #             y2 = min(self.bbox[3] + margin, self.frame_size[0])  # bbox[3] is y2
    #         # negative ty - objects are moving down
    #         else:
    #             x1 = max(self.bbox[0] - tx, 0)  # bbox[0] is x1
    #             x2 = min(self.bbox[2] + margin, self.frame_size[1])  # bbox[2] is x2
    #             y1 = max(self.bbox[1] - margin, 0)  # bbox[1] is y1
    #             y2 = min(self.bbox[3] - ty, self.frame_size[0])  # bbox[3] is y2
    #     # negative tx - objects are moving right
    #     else:
    #         # positive ty - objects are moving up
    #         if ty > 0:
    #             x1 = max(self.bbox[0] - margin, 0)  # bbox[0] is x1
    #             x2 = min(self.bbox[2] - tx, self.frame_size[1])  # bbox[2] is x2
    #             y1 = max(self.bbox[1] - ty, 0)  # bbox[1] is y1
    #             y2 = min(self.bbox[3] + margin, self.frame_size[0])  # bbox[3] is y2
    #         # negative ty - objects are moving down
    #         else:
    #             x1 = max(self.bbox[0] - margin, 0)  # bbox[0] is x1
    #             x2 = min(self.bbox[2] - tx, self.frame_size[1])  # bbox[2] is x2
    #             y1 = max(self.bbox[1] - margin, 0)  # bbox[1] is y1
    #             y2 = min(self.bbox[3] - ty, self.frame_size[0])  # bbox[3] is y2
    #
    #
    #     return x1, y1, x2, y2

    def add(self, det, depth, id_, frame_size):
        self.track_id = id_
        self.bbox = [det[0], det[1], det[2], det[3]]
        self.is_activated = True
        self.score = det[4] * det[5]
        self.cls = det[6]
        self._count += 1
        self.frame_size = frame_size
        self.depth = depth
        #self.state = TrackState.New



    def update(self, det, depth):
        bbox = [det[0], det[1], det[2], det[3]]
        self.accumulated_dist.append(self.bbox[0] - bbox[0])
        self.accumulated_height.append(self.bbox[1] - bbox[1])
        if self.accumulated_dist.__len__() > 3:
            self.accumulated_dist.pop(0)
        if self.accumulated_height.__len__() > 3:
            self.accumulated_height.pop(0)
        self.bbox = bbox
        self.is_activated = True
        self.score = det[4] * det[5]
        self.cls = det[6]
        self._count += 1
        #self.state = TrackState.Tracked
        self.lost_counter = 0
        if not isinstance(depth, type(None)):
            if not np.isnan(depth):
                self.depth = depth
    def output(self):
        return [self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3], self.score, self.cls, self.track_id]






