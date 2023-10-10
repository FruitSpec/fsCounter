import os
import csv
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import collections
from collections.abc import Iterable

from vision.visualization.drawer import draw_rectangle, draw_text, draw_highlighted_test, get_color
from vision.depth.zed.svo_operations import get_dimensions
from vision.misc.help_func import validate_output_path, load_json, write_json, read_json
from vision.depth.slicer.slicer_flow import post_process
from vision.tools.video_wrapper import video_wrapper



class ResultsCollector():

    def __init__(self, rotate=False, mode=""):

        self.detections = []
        self.detections_header = ["x1", "y1", "x2", "y2", "obj_conf", "class_conf", "class_pred", "frame_id"]
        self.tracks = []
        self.tracks_header = ["x1", "y1", "x2", "y2", "obj_conf", "class_conf", "track_id", "frame_id", "depth"]
        self.results = []
        self.file_names = []
        self.file_ids = []
        self.rotate = rotate
        self.alignment = []
        self.alignment_header = ["x1", "y1", "x2", "y2", "tx", "ty", "frame", "zed_shift"]
        self.jai_translation = []
        self.jai_translation_header = ["tx", "ty", "frame"]
        self.jai_zed = {}
        self.trees = {}
        self.hash = {}
        self.jai_width = 1536
        self.jai_height = 2048
        self.percent_seen = []
        self.mode = mode


    def collect_adt(self, trk_outputs, alignment_results, percent_seen, f_id, jai_translation_results=[]):
        self.collect_tracks(trk_outputs)
        self.collect_alignment(alignment_results, f_id)
        if not isinstance(percent_seen, type(None)):
            self.collect_percent_seen(percent_seen, f_id)
        self.collect_jai_translation(jai_translation_results, f_id)

    def collect_jai_translation(self, jai_translation_results, f_id):
        if not jai_translation_results:
            return
        for i, translations in enumerate(jai_translation_results):
            self.jai_translation.append(list(translations) + [f_id+i])

    def collect_percent_seen(self, percent_seen, f_id):
        if isinstance(percent_seen, Iterable):
            for i, frame_percent_seen in enumerate(percent_seen):
                self.percent_seen.append([f_id + i] + list(frame_percent_seen))
        else:
            self.percent_seen.append([f_id] + list(percent_seen))


    def collect_detections(self, batch_results, img_id):
        for i, detection_results in enumerate(batch_results):
            if detection_results is not None:
                output = []
                for det in detection_results:
                    det.append(img_id + i)
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

    def collect_tracks(self, batch_results):

        for tracking_results in batch_results:
            self.tracks += tracking_results

        #return tracking_results

    def collect_results(self, tracking_results, clusters, dimentsions, colors):

        results = []
        for i in range(len(tracking_results)):
            temp = tracking_results[i]
            temp.append(clusters[i])
            temp += dimentsions[i]
            temp += colors[i]

            results.append(temp)

        self.results += results

        return results
    def collect_size_measure(self, point_cloud_mat, tracking_results):
        self.measures += get_dimentions(point_cloud_mat, tracking_results)

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

    def get_row_fileds_tracks(self):
        if self.mode in ["", "tomato"]:
            fields = ["x1", "y1", "x2", "y2", "obj_conf", "class_pred", "track_id", "frame"]
        elif self.mode in ["depth", "depth_tomato"]:
            fields = ["x1", "y1", "x2", "y2", "obj_conf", "class_pred", "track_id", "frame", "depth"]
        elif self.mode in ["pc", "pc_tomato"]:
            fields = ["x1", "y1", "x2", "y2", "obj_conf", "class_pred", "track_id", "frame", "pc_x", "pc_y",
                      "depth"]
        else:
            fields = ["x1", "y1", "x2", "y2", "obj_conf", "class_pred", "track_id", "frame", "pc_x", "pc_y",
                      "depth", "width", "height"]
        if self.mode.endswith("tomato"):
            fields += ["color"]
        rows = self.tracks
        return rows, fields

    def get_row_fields_dets(self):
        n_fileds = len(self.detections[0])
        if self.mode == "":
            fields = ["x1", "y1", "x2", "y2", "obj_conf", "class_conf", "class_pred", "image_id"]
        elif self.mode == "depth":
            fields = ["x1", "y1", "x2", "y2", "obj_conf", "class_conf", "class_pred", "depth", "image_id"]
        elif self.mode == "pc":
            fields = ["x1", "y1", "x2", "y2", "obj_conf", "class_conf", "class_pred", "pc_x", "pc_y", "depth",
                      "image_id"]
        else:
            fields = ["x1", "y1", "x2", "y2", "obj_conf", "class_conf", "class_pred", "pc_x", "pc_y", "depth",
                      "width", "height", "image_id"]
        return self.detections, fields

    def dump_to_csv(self, output_file_path, type='detections'):
        if type == 'detections':
            rows, fields = self.get_row_fields_dets()
        elif type == "jai_translations":
            fields = self.jai_translation_header
            rows = self.jai_translation
        elif type == 'measures':
            fields = ["x1", "y1", "x2", "y2", "obj_conf", "class_conf", "track_id", "frame", "cluster", "height",
                      "width", "color", "color_std"]
            rows = self.results
        elif type == "alignment":
            fields = self.alignment_header
            rows = self.alignment
        elif type == "percen_seen":
            fields = ["frame", "percent_seen", "percent_h_seen", "percent_seen_top", "no_tree_indicator", "full_tree"]
            rows = self.percent_seen
        else:
            rows, fields = self.get_row_fileds_tracks()
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
        tot_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        if not write_frames:
            width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            fps = cap.get(cv2.CAP_PROP_FPS)
            output_video_name = os.path.join(output_path, 'result_video.mp4')
            output_video = cv2.VideoWriter(output_video_name, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                           fps, (width, height))
        # Read until video is completed
        print("writing results")
        ids_in_hash = list(hash.keys())
        f_id = 0

        pbar = tqdm(total=tot_frames)
        while (cap.isOpened()):

            ret, frame = cap.read()
            if ret == True:
                pbar.update(1)
                if f_id in ids_in_hash:
                    dets = hash[f_id]
                else:
                    dets = []
                if self.rotate:
                    frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                if write_frames:
                    self.draw_and_save(frame, dets, f_id, output_path)
                else:
                    frame = self.draw_dets(frame, dets)
                    output_video.write(frame)

                f_id += 1
            else:
                break

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
        for i, id_ in tqdm(enumerate(self.file_ids)):
            if id_ in img_with_dets:
                dets = hash_[id_]
            else:
                dets = []

            frame = cv2.imread(os.path.join(data_dir, self.file_names[i]))
            #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            self.draw_and_save(frame, dets, id_, output_path)

    def draw_and_save(self, frame, dets, f_id, output_path, t_index=6, det_colors=None):

        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = self.draw_dets(frame, dets, t_index=t_index, det_colors=det_colors)
        output_file_name = os.path.join(output_path, f'frame_{f_id}_res.jpg')
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_file_name, frame)

    def draw_and_save_batch(self, batch_frame, batch_dets, f_id, output_path, t_index=6):

        for id_ in range(len(batch_frame)):
            self.draw_and_save(batch_frame[id_], batch_dets[id_], f_id + id_, output_path)

    @staticmethod
    def draw_dets(frame, dets, t_index=6, text=True, det_colors=None):

        for i, det in enumerate(dets):
            track_id = det[t_index]
            color_id = int(track_id) % 15  # 15 is the number of colors in list
            if det_colors is not None:
                color = det_colors[i]
            else:
                color = get_color(color_id)
            text_color = get_color(-1)
            frame = draw_rectangle(frame, (int(det[0]), int(det[1])), (int(det[2]), int(det[3])), color, 3)
            if text:
                frame = draw_highlighted_test(frame, f'ID:{int(track_id)}', (det[0], det[1]), frame.shape[1], color, text_color,
                                              True, 10, 3)

        return frame

    @staticmethod
    def create_hash(detections):

        hash = {}
        id_list = []
        for det in detections:
            if det[-1] in id_list:  # -1 is image id
                hash[det[-1]].append(det)
            else:
                hash[det[-1]] = [det]
                id_list.append(det[-1])

        return hash

    def debug_batch(self, batch_id, args, trk_outputs, det_outputs, frames, depth=None, trk_windows=None,
                    det_colors=None, zed_frames=None, alignment_results=None):
        for i in range(len(trk_outputs)):
            f_id = batch_id + i
            f_depth = depth[i] if depth is not None else None
            f_windows = trk_windows[i] if trk_windows is not None else None
            f_det_colors = det_colors[i] if det_colors is not None else None
            zed_frame = zed_frames[i] if zed_frames is not None else None
            f_alignment_results = alignment_results[i] if alignment_results is not None else None
            self.debug(f_id, args, trk_outputs[i], det_outputs[i], frames[i], depth=f_depth, trk_windows=f_windows,
                       det_colors=f_det_colors, zed_frame=zed_frame, alignment_results=f_alignment_results)

    def debug(self, f_id, args, trk_outputs, det_outputs, frame, depth=None, trk_windows=None, det_colors=None,
              zed_frame=None, alignment_results=None):
        if args.debug.tracker_windows and trk_windows is not None:
            self.save_tracker_windows(f_id, args, trk_outputs, trk_windows)
        if args.debug.tracker_results:
            validate_output_path(os.path.join(args.output_folder, 'trk_results'))
            self.draw_and_save(frame.copy(), trk_outputs, f_id, os.path.join(args.output_folder, 'trk_results'),
                               det_colors=det_colors)
        if args.debug.det_results:
            validate_output_path(os.path.join(args.output_folder, 'det_results'))
            self.draw_and_save(frame.copy(), det_outputs, f_id, os.path.join(args.output_folder, 'det_results'),
                               det_colors=det_colors)
        if args.debug.raw_frame:
            validate_output_path(os.path.join(args.output_folder, 'frames'))
            self.draw_and_save(frame.copy(), [], f_id, os.path.join(args.output_folder, 'frames'),
                               det_colors=det_colors)
        if args.debug.depth and depth is not None:
            validate_output_path(os.path.join(args.output_folder, 'depth'))
            self.draw_and_save(depth.copy(), [], f_id, os.path.join(args.output_folder, 'depth'),
                               det_colors=det_colors)
        if args.debug.clusters:
            validate_output_path(os.path.join(args.output_folder, 'clusters'))
            self.draw_and_save(frame.copy(), trk_outputs, f_id, os.path.join(args.output_folder, 'clusters'), -5,
                               det_colors=det_colors)
        if args.debug.alignment:
            validate_output_path(os.path.join(args.output_folder, 'alignment'))
            save_aligned(zed_frame, frame, args.output_folder, f_id, corr=alignment_results[0], sub_folder='alignment',
                         dets=trk_outputs)
    @staticmethod
    def save_tracker_windows(f_id, args, trk_outputs, trk_windows):
        canvas = np.zeros((args.frame_size[0], args.frame_size[1], 3)).astype(np.uint8)
        for w in trk_windows:
            canvas = cv2.rectangle(canvas, (int(w[0]), int(w[1])), (int(w[2]), int(w[3])), (255, 0, 0),
                                   thickness=-1)
        for t in trk_outputs:
            canvas = cv2.rectangle(canvas, (int(t[0]), int(t[1])), (int(t[2]), int(t[3])), (0, 0, 255),
                                   thickness=-1)
        canvas = cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR)

        validate_output_path(os.path.join(args.output_folder, 'windows'))
        cv2.imwrite(os.path.join(args.output_folder, 'windows', f"windows_frame_{f_id}.jpg"), canvas)

    def collect_alignment(self, alignment_results, f_id):
        for i, r in enumerate(alignment_results):
            x1, y1, x2, y2 = r[0]
            tx = r[1]
            ty = r[2]
            zed_shift = r[3]
            tx = tx if not np.isnan(tx) else 0
            ty = ty if not np.isnan(ty) else 0
            self.alignment.append([x1, y1, x2, y2, int(tx), int(ty), f_id + i, zed_shift])
            self.jai_zed[f_id + i] = (f_id + i) + zed_shift

        pass

    def dump_to_trees(self, output_path, sliced_data, save=True):
        hash = self.create_frame_to_trees_hash(sliced_data)
        hash_ids = list(hash.keys())
        trees = {}

        for track in self.tracks:
            frame_id = track[-1]
            if not frame_id in hash_ids:
                continue
            cur_trees = hash[frame_id]
            for tree in cur_trees:
                tree_ids = list(trees.keys())
                if tree in tree_ids:
                    trees[tree].append(track)
                else:
                    trees[tree] = [track]

        # saved for later use of other results dumps
        self.hash = hash
        self.trees = trees
#        if save:
#            write_json(os.path.join(output_path, "trees_det.json"), trees)

    def convert_tracker_results(self, frames):
        """
        ["x1", "y1", "x2", "y2", "obj_conf", "class_conf", "track_id", "frame"]
        filters the tracker results to given frames and converts to feature extractor moudle format
        :param frames: frames for tree
        :return: trakcer results in new format and old format
        """
        tracker_full_results = np.array(self.tracks)
        frames_tracker_results = tracker_full_results[np.isin(tracker_full_results[:, 7], frames), :]
        tracker_results = {}
        for i, frame in enumerate(frames):
            tracker_results[frame] = {int(row[6]): ((int(row[0]), int(row[1])), (int(row[2]), int(row[3]))) for row in frames_tracker_results}
        return tracker_results, frames_tracker_results

    def get_pc_for_jai(self, frame, zed_cam, aligemnet_df):
        """
        cuts a point cloud to jai_in_zed_coords and resizes it to jai size
        this is a preprocess phase so that the point cloud and jai frame will be aligned
        :param frame: int! indicating the frame number of jai image
        :param zed_cam: zed camera video_wrapper object
        :param aligemnet_df: alignment dataframe
        :return: point cloud aligned to jai
        """
        point_cloud = zed_cam.get_zed(self.jai_zed[str(frame)])[2]
        point_cloud_shape = point_cloud.shape
        x1, y1, x2, y2 = aligemnet_df[aligemnet_df["frame"] == frame].values[0].astype(int)[:4]
        x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, point_cloud_shape[0]-1), min(y2, point_cloud_shape[1]-1)
        point_cloud_for_jai = point_cloud[y1:y2, x1:x2]
        return cv2.resize(point_cloud_for_jai, (self.jai_width, self.jai_height))


    def save_filtered_cv(self, output_path, sliced_df, zed_cam, max_z=5):
        """
        will calculate cv per tree with depth filtering
        :param output_path: where to write the files to
        :param sliced_df: a dataframe containing slicing data
        :param zed_cam: zed video wrapper object
        :param max_z: max allowed distance
        :return:
        """
        trees = sliced_df["tree_id"].unique()
        slicer_results = get_slicer_data(sliced_df, self.jai_width, tree_id=-1)
        aligemnet_df = pd.DataFrame(self.alignment, columns=self.alignment_header)
        cvs = []
        res_df = pd.DataFrame([])
        for tree in tqdm(trees):
            frames = sliced_df[sliced_df["tree_id"] == tree]["frame_id"].values.astype(int).tolist()
            tracker_results, frames_tracker_results = self.convert_tracker_results(frames)
            slicer_results_tree = {frame: slicer_results[str(frame)] for frame in frames}
            tree_images = {frame: {"zed": self.get_pc_for_jai(frame, zed_cam, aligemnet_df), "nir": None, "swir975": None
                                   } for frame in frames}
            tracker_results = filter_tracker_results(tracker_results, slicer_results_tree, tree_images, max_z)
            cvs.append(tracker_results["cv"])
            tracker_frame = pd.DataFrame(frames_tracker_results, columns=['x1', 'y1', 'x2', 'y2', "score", "class",
                                                                          "track_id", "frame_id"])
            tracker_frame["tree_id"] = tree
            res_df = pd.concat([res_df, tracker_frame])
        df = pd.DataFrame({'tree_id': trees, 'cv': cvs})
        df.to_csv(os.path.join(output_path, 'trees_cv.csv'))

        res_df.to_csv(os.path.join(output_path, 'trees_sliced_track.csv'))
        return df

    def save_trees_sliced_track(self, output_path, sliced_df):
        """
        will create trees_sliced_track dataframe
        :param output_path: where to write the files to
        :param sliced_df: a dataframe containing slicing data
        :return:
        """
        trees = sliced_df["tree_id"].unique()
        res_df = pd.DataFrame({})
        for tree in tqdm(trees):
            frames = sliced_df[sliced_df["tree_id"] == tree]["frame_id"].values.astype(int).tolist()
            tracker_results, frames_tracker_results = self.convert_tracker_results(frames)
            n_fileds = len(frames_tracker_results[0])
            if n_fileds == 8:
                columns = ["x1", "y1", "x2", "y2", "obj_conf", "class_pred", "track_id", "frame"]
            elif n_fileds == 9:
                columns = ["x1", "y1", "x2", "y2", "obj_conf", "class_pred", "track_id", "frame", "depth"]
            elif n_fileds == 11:
                columns = ["x1", "y1", "x2", "y2", "obj_conf", "class_pred", "track_id", "frame", "pc_x", "pc_y",
                          "depth"]
            else:
                columns = ["x1", "y1", "x2", "y2", "obj_conf", "class_pred", "track_id", "frame", "pc_x", "pc_y",
                          "depth", "width", "height"]
            tracker_frame = pd.DataFrame(frames_tracker_results, columns=columns)
            tracker_frame["tree_id"] = tree
            res_df = pd.concat([res_df, tracker_frame])

        res_df.to_csv(os.path.join(output_path, 'trees_sliced_track.csv'))
        return res_df

    def dump_to_cv(self, output_path, sliced_df):

        if len(self.trees.keys()) == 0:
            self.dump_to_trees("", sliced_df, save=False)

        filtered_trees = {}
        trees = list(self.trees.keys())

        for tree in trees:
            filtered_tree_data = []
            tree_data = self.trees[tree].copy()
            tree_slice = sliced_df[sliced_df['tree_id'] == tree]

            for track in tree_data:
                frame_slice_data = tree_slice[tree_slice['frame_id'] == track[-1]]
                if frame_slice_data['start'].values[0] == -1 and frame_slice_data['end'].values[0] == -1:
                    filtered_tree_data.append(track)
                elif frame_slice_data['start'].values[0] == -1:  # only end exist
                    if frame_slice_data['end'].values[0] > track[0]:
                        filtered_tree_data.append(track)
                elif frame_slice_data['end'].values[0] == -1:  # only start exist
                    if frame_slice_data['start'].values[0] < track[2]:
                        filtered_tree_data.append(track)
                else:
                    if frame_slice_data['start'].values[0] < track[2] and frame_slice_data['end'].values[0] > track[0]:
                        filtered_tree_data.append(track)

            filtered_trees[tree] = filtered_tree_data

        tree_cv = self.filtered_trees_to_cv(filtered_trees)

        df = pd.DataFrame(data=tree_cv, columns=['tree_id', 'cv'])
        df.to_csv(os.path.join(output_path, 'trees_cv.csv'))

        res = []
        filtered_trees_ids = list(filtered_trees.keys())
        for tree in filtered_trees_ids:
            tree_data = filtered_trees[tree]
            for track in tree_data:
                res.append({"x1": track[0],
                            "y1": track[1],
                            "x2": track[2],
                            "y2": track[3],
                            "score": track[4],
                            "class": track[5],
                            "track_id": track[6],
                            "frame_id": track[7],
                            "tree_id": tree}
                           )
        res_df = pd.DataFrame(data=res, columns=['x1', 'y1', 'x2', 'y2', "score", "class",
                                                 "track_id", "frame_id", "tree_id"])
        res_df.to_csv(os.path.join(output_path, 'trees_sliced_track.csv'))

        #write_json(os.path.join(output_path, 'trees_sliced_track.json'), filtered_trees)
        return filtered_trees

    @staticmethod
    def filtered_trees_to_cv(filtered_trees):
        trees = list(filtered_trees.keys())
        tree_cv = []
        for tree in trees:
            tree_ids = []
            tree_tracks = filtered_trees[tree]
            for track in tree_tracks:
                if track[-2] not in tree_ids: # add only once every id in tree
                    tree_ids.append(track[-2])
            tree_cv.append({'tree_id': tree, 'cv': len(tree_ids)})

        return tree_cv

    def save_det_images(self, output_path, filtered_trees, jai_movie_path, jai_rotate):

        jai_cam = video_wrapper(jai_movie_path, jai_rotate)

        # Read until video is completed
        print(f'writing results on {jai_movie_path}\n')
        tree_ids = list(filtered_trees.keys())
        pbar = tqdm(total=len(tree_ids))
        for tree in tree_ids:
            tree_output_path = os.path.join(output_path, f"T{tree}")
            validate_output_path(tree_output_path)
            pbar.update(1)
            index = -1
            tree_data = filtered_trees[tree]
            cur_frame_dets = []
            for track in tree_data:
                if track[-1] != index:
                    if len(cur_frame_dets) > 0:
                        self.draw_and_save(jai_frame, cur_frame_dets, int(index), tree_output_path)
                    index = track[-1]
                    fsi_ret, jai_frame = jai_cam.get_frame(index)

                    if not fsi_ret:  # couldn't get frames
                        # Break the loop
                        break
                    cur_frame_dets = [track]
                else:
                    cur_frame_dets.append(track)

        jai_cam.close()


    def dump_state(self, output_path):
        write_json(os.path.join(output_path, 'alignment.json'), self.alignment)

    def dump_cv_res(self, output_path: str, depth: int = 0) -> None:
        """
        Saves to csv the cv results after min tracks and depth filtering operations
        Args:
            output_path (str): output folder path
            depth (int): max depth for fruits to be counted (0 means no depth filtering)

        Returns:

        """
        block_folder = os.path.dirname(output_path)
        row = os.path.basename(output_path)
        rows, fields = self.get_row_fileds_tracks()
        track_df = pd.DataFrame(rows, columns=fields)
        block = block_folder.split('/')[-1]
        cvs_min_samp = {f"n_unique_track_ids_{i}": [] for i in range(2, 6)}
        cvs_min_samp_filtered = {f"n_unique_track_ids_filtered_{depth}_{i}": [] for i in range(2, 6)}
        track_list = [track_df["track_id"].nunique()]
        uniq, counts = np.unique(track_df["track_id"], return_counts=True)
        for i in range(2, 6):
            cvs_min_samp[f"n_unique_track_ids_{i}"].append(len(uniq[counts >= i]))
        if "depth" in track_df.columns:
            track_df = track_df[track_df["depth"] < depth]
            uniq, counts = np.unique(track_df["track_id"], return_counts=True)
            track_list_filtered = [track_df["track_id"].nunique()]
            for i in range(2, 6):
                cvs_min_samp_filtered[f"n_unique_track_ids_filtered_{depth}_{i}"].append(len(uniq[counts >= i]))
        final_csv_path = os.path.join(output_path, f'{block}_{row}_n_track_ids.csv')
        final_csv_path_filtered = os.path.join(output_path, f'{block}_{row}_n_track_ids_filtered_{depth}.csv')
        pd.DataFrame({"row": [row], "n_unique_track_ids": track_list, **cvs_min_samp}).to_csv(final_csv_path)
        pd.DataFrame({"row": [row],
                      f"n_unique_track_ids_filtered_{depth}": track_list_filtered, **cvs_min_samp_filtered}) \
            .to_csv(final_csv_path_filtered)


    def dump_feature_extractor(self, output_path, depth=0):
        """
        this function is a wrapper for dumping data to files for the feature extraction pipline
        :param output_path: where to output to
        :param depth: max depth for fruits to be counted (0 means no depth filtering)
        :return: None
        """
        # self.dump_state(output_path)
        self.dump_to_csv(os.path.join(output_path, 'tracks.csv'), type="tracks")
        self.dump_to_csv(os.path.join(output_path, 'alignment.csv'), type="alignment")
        self.dump_to_csv(os.path.join(output_path, 'jai_translations.csv'), type="jai_translations")
        self.dump_to_csv(os.path.join(output_path, 'percen_seen.csv'), type="percen_seen")
        # self.dump_cv_res(output_path, depth)

    def converted_slice_data(self, sliced_data):
        converted_sliced_data = {}

        zed_frame_to_coor = self.alignment_hash()
        zed_frame_to_coor_keys = list(zed_frame_to_coor.keys())
        zed_frame_ids = list(sliced_data.keys())
        zed_jai_hash = self.convert_to_zed_to_jai(self.jai_zed)
        zed_jai_hash_keys = list(zed_jai_hash.keys())
        for zed_frame_id in zed_frame_ids:
            if not ((zed_frame_id in zed_jai_hash_keys) and (zed_frame_id in zed_frame_to_coor_keys)):
                continue
            jai_frame_id = zed_jai_hash[zed_frame_id]
            zed_coor = zed_frame_to_coor[zed_frame_id]
            slice_coor = sliced_data[zed_frame_id]
            tx = zed_coor[0]
            if len(slice_coor) == 0:
                continue
            factor = self.jai_width / (zed_coor[1] - zed_coor[0])
            new_slice_cor = []
            for slice in slice_coor:
                if slice < zed_coor[0]:  # outside roi
                    new_slice_cor.append(10)  # put slice at frame start
                elif slice > zed_coor[1]:
                    new_slice_cor.append(self.jai_width - 10) # put slice at frame end
                else:
                    new_slice_cor.append((slice - tx) * factor)
            converted_sliced_data[jai_frame_id] = new_slice_cor

        return converted_sliced_data



    @staticmethod
    def convert_to_zed_to_jai(jai_zed_hash):
        zed_jai_hash = {}
        jai_frame_ids = list(jai_zed_hash.keys())

        for j_f in jai_frame_ids:
            zed_frame_id = jai_zed_hash[j_f]
            zed_jai_hash[zed_frame_id] = j_f

        zed_jai_hash = collections.OrderedDict(sorted(zed_jai_hash.items()))

        return zed_jai_hash



    @staticmethod
    def create_frame_to_trees_hash(sliced_data):
        hash = {}
        frame_list = list(sliced_data['frame_id'])
        for frame_id in frame_list:
            sub = sliced_data[sliced_data['frame_id'] == frame_id]
            hash[frame_id] = list(pd.unique(sub['tree_id']))

        return hash

    def alignment_hash(self):

        hash = {}
        for res in self.alignment:
            hash[res['frame']] = [res['x1'], res['x2']]

        return hash

    def set_alignment(self, alignment):
        """
        sets alignment argument from out source data
        :param alignment: path or file, if path will read json from path, else will set file
        :return: None
        """
        if isinstance(alignment, str):
            self.alignment = read_json(alignment)
        else:
            self.alignment = alignment

    def set_jai_zed(self, jai_zed):
        """
        sets jai_zed argument from out source data
        :param jai_zed: path or file, if path will read json from path, else will set file
        :return: None
        """
        if isinstance(jai_zed, str):
            self.jai_zed = read_json(jai_zed)
        else:
            self.jai_zed = jai_zed

    def set_detections(self, detections):
        """
        sets detections argument from out source data
        detections should be a list of list, and keeps the original data type [int,int,int,int,float,float,float,int]
        :param detections: path or file, if path will read dataframe from path, else will set file
        :return: None
        """
        if isinstance(detections, str):
            detections = pd.read_csv(detections)
            rows_dict = detections.to_dict(orient="records")
            self.detections = [[row[col] for col in detections.columns] for row in rows_dict]
        else:
            self.detections = detections

    def set_tracks(self, tracks):
        """
        sets detections argument from out source data
        tracks should be a list of list, and keeps the original data type [int,int,int,int,float,float,int,int]
        :param tracks: path or file, if path will read dataframe from path, else will set file
        :return: None
        """
        if isinstance(tracks, str):
            tracks = pd.read_csv(tracks)
            rows_dict = tracks.to_dict(orient="records")
            self.tracks = [[row[col] for col in tracks.columns] for row in rows_dict]
        else:
            self.tracks = tracks

    def set_percent_seen(self, percent_seen):
        """
        sets percent_seen argument from out source data
        :param percent_seen: path or file, if path will read json from path, else will set file
        :return: None
        """
        if isinstance(percent_seen, str):
            self.percent_seen = pd.read_csv(percent_seen).tolist()
        else:
            self.percent_seen = percent_seen

    def set_self_params(self, read_from, parmas=["alignment", "jai_zed", "detections", "tracks"]):
        """
        sets the paramaters from out source data
        :param read_from: folder to read from
        :param parmas: paramaters to set
        :return:
        """
        if "alignment" in parmas:
            self.set_alignment(os.path.join(read_from, 'alignment.json')) # list of dicts
        if "jai_zed" in parmas:
            self.set_jai_zed(os.path.join(read_from, 'jai_zed.json')) # dict
        if "detections" in parmas:
            self.set_detections(os.path.join(read_from, 'detections.csv')) # list of lists
        if "tracks" in parmas:
            self.set_tracks(os.path.join(read_from, 'tracks.csv')) # list of lists
        if "percent_seen" in parmas:
            self.set_percent_seen(os.path.join(read_from, "percent_seen.csv"))


def scale(det_dims, frame_dims):
    r = min(det_dims[0] / frame_dims[0], det_dims[1] / frame_dims[1])
    return (1 / r)


def scale_det(detection, scale_):

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

def filter_tracker_results(tracker_results, slicer_results, tree_images, max_z):
    """
    filters the tracker results based on slicing and max depth allowed
    :param tracker_results: tracker results dictionary
    :param slicer_results: slicer results dictionary
    :param tree_images: tree images dictionary
    :param max_z: max depth allowed
    :return: filter dictionary
    """
    tracker_results = filter_outside_tree_boxes(tracker_results, slicer_results)
    if max_z > 0:
        tracker_results = filter_outside_zed_boxes(tracker_results, tree_images, max_z)
    tracker_results["cv"] = len(
        {id for frame in set(tracker_results.keys()) - {"cv"} for id in tracker_results[frame].keys()})
    return tracker_results

def filter_outside_zed_boxes(tracker_results, tree_images, max_z):
    """
    removes detections that are too far
    :param tracker_results: dict of the tracker results
    :param slicer_results: dict of the slicer results
    :param max_z: maxsimum depth allowed
    :return: updated tracker_results
    """
    for frame in tree_images.keys():
        frame_images = tree_images[frame]
        boxes = tracker_results[frame]
        to_pop = []
        for id, box in boxes.items():
            t, b, l, r = box[0][1], box[1][1], box[0][0], box[1][0]
            z = np.nanmean(frame_images["zed"][t:b, l:r, 2])
            if z > max_z or np.isnan(z):
                to_pop.append(id)
        for id in to_pop:
            boxes.pop(id)
    return tracker_results

def filter_outside_tree_boxes(tracker_results, slicer_results):
    """
    removes detections that are not on the tree
    :param tracker_results: dict of the tracker results
    :param slicer_results: dict of the slicer results
    :return: updated tracker_results
    """
    for frame in tracker_results.keys():
        if frame == "cv":
            continue
        x_start, x_end = slicer_results[frame]
        x_0 = np.array([box[0][0] for box in tracker_results[frame].values()])
        x_1 = np.array([box[1][0] for box in tracker_results[frame].values()])
        for id in np.array(list(tracker_results[frame].keys()))[np.all([x_0 > x_start, x_1 < x_end], axis=0) == False]:
            tracker_results[frame].pop(id)
    return tracker_results
def get_slicer_data(slice_path, max_w, tree_id=-1):
    """
    reads slicer data
    :param slice_path: path to slices.csv
    :param max_w: max width of picture
    :param tree_id: if -1 will use old pipe, else will use new pipe
    :return: dataframe contaiting the slices data
    """
    if isinstance(slice_path, str):
        sliced_data = pd.read_csv(slice_path)
    else:
        sliced_data = slice_path
    if tree_id != -1:
        sliced_data = sliced_data[sliced_data["tree_id"] == tree_id]
    if "starts" in sliced_data.keys():
        sliced_data["start"] = sliced_data["starts"]
    if "ends" in sliced_data.keys():
        sliced_data["end"] = sliced_data["ends"]
    sliced_data["start"].replace(-1, 0, inplace=True)
    sliced_data["end"].replace(-1, max_w, inplace=True)
    sliced_data = dict(zip(sliced_data["frame_id"].apply(str),
                           tuple(zip(sliced_data["start"].apply(int), sliced_data["end"].apply(int)))))
    return sliced_data


def save_loaded_images(zed_frame, jai_frame, f_id, output_fp, crop=[], draw_rec=False):
    if len(crop) > 0:
        x1 = crop[0]
        y1 = crop[1]
        x2 = crop[2]
        y2 = crop[3]
        if not draw_rec:
            cropped = zed_frame[int(y1):int(y2), int(x1):int(x2)]
            cropped = cv2.resize(cropped, (480, 640))
        else:
            cropped = zed_frame.copy()
            cropped = cv2.rectangle(cropped, (x1, y1), (x2, y2), (255, 0, 0), 3)
            cropped = cv2.resize(cropped, (480, 640))
    else:
        cropped = cv2.resize(zed_frame, (480, 640))

    jai_frame = cv2.resize(jai_frame, (480, 640))

    canvas = np.zeros((700, 1000, 3), dtype=np.uint8)
    canvas[10:10 + cropped.shape[0], 10:10 + cropped.shape[1], :] = cropped
    canvas[10:10 + jai_frame.shape[0], 510:510 + jai_frame.shape[1], :] = jai_frame

    file_name = os.path.join(output_fp, f"frame_{f_id}.jpg")
    cv2.imwrite(file_name, canvas)


def save_aligned(zed, jai, output_folder, f_id, corr=None, sub_folder='FOV', dets=None):
    if corr is not None and np.sum(np.isnan(corr)) == 0:
        zed = zed[int(corr[1]):int(corr[3]), int(corr[0]):int(corr[2]), :]

    gx = 680 / jai.shape[1]
    gy = 960 / jai.shape[0]
    zed = cv2.resize(zed, (680, 960))
    jai = cv2.resize(jai, (680, 960))

    if dets is not None:
        dets = np.array(dets)
        dets[:, 0] = dets[:, 0] * gx
        dets[:, 2] = dets[:, 2] * gx
        dets[:, 1] = dets[:, 1] * gy
        dets[:, 3] = dets[:, 3] * gy
        jai = ResultsCollector.draw_dets(jai, dets, t_index=6, text=False)
        zed = ResultsCollector.draw_dets(zed, dets, t_index=6, text=False)

    canvas = np.zeros((960, 680 * 2, 3))
    canvas[:, :680, :] = zed
    canvas[:, 680:, :] = jai

    fp = os.path.join(output_folder, sub_folder)
    validate_output_path(fp)
    cv2.imwrite(os.path.join(fp, f"aligned_f{f_id}.jpg"), canvas)



if __name__ == "__main__":
    slice_data_path = "/home/fruitspec-lab/FruitSpec/Sandbox/DWDB_2023/DWDBCN51_test/200123/DWDBCN51/R13/ZED_1_slice_data.json"
    output_path = "/home/fruitspec-lab/FruitSpec/Sandbox/DWDB_2023/DWDBCN51_test/200123/DWDBCN51/R13/"
    slice_data = load_json(slice_data_path)
    slice_df = post_process(slice_data=slice_data)

    rc = ResultsCollector()
    rc.create_frame_to_trees_hash(slice_df)