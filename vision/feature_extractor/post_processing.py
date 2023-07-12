import pandas as pd
from vision.feature_extractor.boxing_tools import project_boxes_to_fruit_space_trilateration, project_boxes_to_fruit_space_global
import numpy as np
from omegaconf import OmegaConf
from vision.misc.help_func import get_repo_dir
from vision.feature_extractor.tree_size_tools import safe_nanmean
from vision.feature_extractor.stat_tools import compute_density_mst
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt


class PostProcessor:
    def __init__(self, args):
        self.tracks = args.tracks
        if isinstance(self.tracks, str):
            self.tracks = pd.read_csv(self.tracks)
        self.features = {}
        self.max_depth = args.max_depth
        self.min_samples = args.min_samples
        if self.max_depth > 0 and "depth" in self.tracks.columns:
            self.tracks = self.tracks[self.tracks["depth"] < self.max_depth]
        self.frames = np.unique(self.tracks["frame"]).astype(int)
        self.track_ids = np.unique(self.tracks["track_id"]).astype(int)
        self.min_dets_3d_init = args.min_dets_3d_init
        self.fruit_3d_space = {}
        self.first_frame = 0
        self.avg_diam = 0
        self.n_avg_diams = args.n_avg_diams
        self.dbscan_min_samples = args.dbscan_min_samples
        self.last_frame_boxes = {}
        self.global_shift = np.array([0, 0, 0])
        self.max_depth_change = args.max_depth_change
        self.filter_tracker_results()

    def get_w_h_ratio_feature(self):
        self.tracks["h"] = self.tracks["y2"] - self.tracks["y1"]
        self.tracks["w"] = self.tracks["x2"] - self.tracks["x1"]
        self.tracks["w_h_ratio"] = self.tracks["w"] / self.tracks["h"]
        backwards = self.tracks["w_h_ratio"] > 1
        self.tracks.loc[backwards, "w_h_ratio"] = 1 / self.tracks["w_h_ratio"][backwards]
        w_h_ratio_grouped = self.tracks.groupby("track_id")["w_h_ratio"]
        wh_std_per_fruit = w_h_ratio_grouped.std()
        n_samp_per_fruit = w_h_ratio_grouped.count()
        self.features["w_h_ratio"] = np.median(wh_std_per_fruit[n_samp_per_fruit > self.min_samples])

    def filter_tracker_results(self):
        if not len(self.tracks):
            return {}
        uniq, counts = np.unique(self.tracks["track_id"], return_counts=True)
        valid_samples = self.tracks["track_id"].isin(uniq[counts >= self.min_samples])
        valid_depths = self.tracks["depth"] < self.max_depth
        self.tracks = self.tracks[valid_depths & valid_samples]

    def init_3d_fruit_space(self):
        n_tracks = self.tracks.groupby("frame")["track_id"].count()
        all_frames = self.frames
        min_dets = np.quantile(n_tracks, self.min_dets_3d_init)
        self.first_frame = all_frames[np.min(np.where(n_tracks > min_dets))]
        first_tracker_results = self.tracks[self.tracks["frame"] == self.first_frame]
        fruit_space = {row[1]["track_id"]: (row[1]["pc_x"], row[1]["pc_y"], row[1]["depth"])
                    for row in first_tracker_results.iterrows()}
        fruits_keys = list(fruit_space.keys())
        for fruit in fruits_keys:
            if np.isnan(fruit_space[fruit][0]):
                fruit_space.pop(fruit)
        return fruit_space

    def create_3d_fruit_space(self):
        self.fruit_3d_space = self.init_3d_fruit_space()
        boxes_w, boxes_h, self.last_frame_boxes = np.array([]), np.array([]), self.fruit_3d_space
        for i, frame_number in enumerate(self.frames):
            if frame_number <= self.first_frame:
                continue
            frame_tracker_results, new_boxes, old_boxes = self.tracks[self.tracks["frame"] == frame_number], {}, {}
            space_keys = self.last_frame_boxes.keys()
            for i, row in frame_tracker_results.iterrows():
                track_id = row["track_id"]
                x_center, y_center, z_center = row["pc_x"], row["pc_y"], row["depth"]
                if track_id not in space_keys:
                    if np.isnan(z_center):
                        continue
                    new_boxes[track_id] = (x_center, y_center, z_center)
                    boxes_w, boxes_h = np.append(boxes_w, row["width"]), np.append(boxes_h, row["height"])
                else:
                    old_boxes[track_id] = (x_center, y_center, z_center)
            if not len(old_boxes):
                print("no old boxes: ", frame_number)
                continue
            self.fruit_3d_space, shift = project_boxes_to_fruit_space_global(self.fruit_3d_space,
                                                                                    self.last_frame_boxes, old_boxes,
                                                                                    new_boxes, self.max_depth_change,
                                                                                    self.global_shift)
            self.global_shift = self.global_shift + shift
            self.last_frame_boxes = dict(zip(frame_tracker_results["track_id"],
                                             tuple(zip(frame_tracker_results["pc_x"],
                                                       frame_tracker_results["pc_y"],
                                                       frame_tracker_results["depth"]))))
        centers = np.array(list(self.fruit_3d_space.values()))
        plot_3d_cloud(self.fruit_3d_space, centers)
        self.avg_diam = np.nanmedian(np.nanmax([boxes_w, boxes_h], axis=0))

    def calc_loc_features(self):
        if not self.fruit_3d_space:
            self.create_3d_fruit_space()
        centers = np.array(list(self.fruit_3d_space.values()))
        problem_w_center = False
        if len(centers) > 0 and len(centers.shape) > 1:
            centers = centers[np.isfinite(centers[:, 2])]
        else:
            problem_w_center = True
        if self.avg_diam > 0 and len(centers) > 2 and not problem_w_center:
            # mst, distances = compute_density_mst(centers)
            clusters = np.unique(DBSCAN(eps=self.avg_diam*self.n_avg_diams,
                                        min_samples=self.dbscan_min_samples).fit(centers).labels_)
            self.features[f"n_clust_arr_{self.n_avg_diams}"] = len(clusters) - 1
        else:
            problem_w_center = True
        # if not problem_w_center:
        #     self.features["mst_mean_arr"] = np.mean(distances)

    def get_features(self):
        self.calc_loc_features()
        self.get_w_h_ratio_feature()
        return self.features

def plot_3d_cloud(fruit_3d_space, centers, c=None):
    """
    Plot the 3D point cloud of the fruit space
    :param fruit_3d_space: Dictionary containing the 3D coordinates of each fruit
    :param centers: array of the coordinates of each fruit
    :param c: color of the points in the point cloud
    :return: None
    """
    fig = plt.figure(figsize=(10, 10))
    ax = plt.axes(projection='3d')
    ax.scatter3D(-centers[:, 2], -centers[:, 0], -centers[:, 1], c=c)
    for i, label in enumerate(fruit_3d_space.keys()):  # plot each point + it's index as text above
        ax.text(-centers[i, 2], -centers[i, 0], -centers[i, 1], '%s' % (str(label)), size=10, zorder=1,
                color='k')
    ax.set_xlabel('z')
    ax.set_ylabel('x')
    ax.set_zlabel('y')
    ax.view_init(20, 20)
    plt.show()
# plot_3d_cloud(self.fruit_3d_space, np.array(list(self.fruit_3d_space.values())))

if __name__ == "__main__":
    repo_dir = get_repo_dir()
    pipeline_config = "/vision/pipelines/config/pipeline_config.yaml"
    runtime_config = "/vision/pipelines/config/dual_runtime_config.yaml"
    cfg = OmegaConf.load(repo_dir + pipeline_config)
    args = OmegaConf.load(repo_dir + runtime_config)
    post_processor = PostProcessor(args.post_processor)
    res = post_processor.get_features()
    print(res)
