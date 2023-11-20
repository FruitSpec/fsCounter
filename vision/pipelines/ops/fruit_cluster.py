import os
import numpy as np
import collections

class Cluster():
    def __init__(self, cluster_id, track_ids):
        self.cluster_id = cluster_id
        self.track_ids = track_ids
        self.lost = 0

class FruitCluster():
    def __init__(self, max_single_fruit_dist=150, range_diff_threshold=0.05, max_losses=5):
        self.clusters = []
        self.cluster_id = 0
        self.max_single_fruit_dist = max_single_fruit_dist
        self.range_diff_threshold = range_diff_threshold
        self.max_losses = max_losses

    def cluster_batch(self, trk_outputs, ranges_batch):
        outputs = []
        clusters_output = []
        for trk_results, ranges in zip(trk_outputs, ranges_batch):
            output, cluster_ids = self.cluster(trk_results, ranges)
            outputs.append(output)
            clusters_output.append(cluster_ids)

        return outputs, clusters_output

    def cluster(self, trk_results, ranges):
        if len(trk_results) == 0:
            return [], []

        trk_results = np.array(trk_results)
        dist = compute_dist_on_vec(trk_results, trk_results)
        diff = compute_diff_on_vec(ranges)

        t_ids = np.array([trk[6] for trk in trk_results])
        neighbors = []
        for t_dist, t_diff in zip(dist, diff):
            dist_bool = t_dist < self.max_single_fruit_dist
            diff_bool = abs(t_diff) < self.range_diff_threshold
            t_id_neighbors = t_ids[dist_bool & diff_bool]
            neighbors.append(t_id_neighbors)

        trk_to_cluster = {}
        live_clusters = []
        for t_id, t_id_neighbors in enumerate(neighbors):
            id_found = False
            for c in self.clusters:
                cluster_found = False
                for n in t_id_neighbors:
                    if n in c.track_ids:
                        id_found = True
                        for t_n in t_id_neighbors:
                            cluster_found = True
                            trk_to_cluster[t_n] = c.cluster_id
                            if t_n in c.track_ids:
                                continue
                            c.track_ids.append(t_n)
                        break
                if cluster_found:
                    if c.cluster_id not in live_clusters:
                        live_clusters.append(c.cluster_id)


                if id_found:
                    break

            if not id_found:
                self.clusters.append(Cluster(self.cluster_id, list(t_id_neighbors)))
                for t_id in list(t_id_neighbors):
                    trk_to_cluster[t_id] = self.cluster_id
                live_clusters.append(self.cluster_id)

                self.cluster_id += 1

        # cleaning
        current_clusters = self.clusters.copy()
        keep_clusters = []
        for cluster in current_clusters:
            if cluster.cluster_id not in live_clusters:
                cluster.lost += 1
            if cluster.lost >= self.max_losses:
                continue
            keep_clusters.append(cluster)

        self.clusters = keep_clusters

        # add class id in results output
        output = trk_results.tolist().copy()
        cluster_ids = []
        for trk in output:
            cluster_id = trk_to_cluster[trk[6]]
            trk.append(cluster_id)
            cluster_ids.append(cluster_id)

        return output, cluster_ids

def compute_dist_on_vec(trk_windows, dets):

    trk_x = (trk_windows[:, 0] + trk_windows[:, 2]) / 2 # - mean_x_movment
    trk_y = (trk_windows[:, 1] + trk_windows[:, 3]) / 2 #- mean_y_movment


    det_x = (dets[:, 0] + dets[:, 2]) / 2
    det_y = (dets[:, 1] + dets[:, 3]) / 2

    trk_x = np.expand_dims(trk_x, axis=0)
    det_x = np.expand_dims(det_x, axis=0)

    sqaured_diff_x = np.power((trk_x.T - det_x), 2)

    trk_y = np.expand_dims(trk_y, axis=0)
    det_y = np.expand_dims(det_y, axis=0)

    sqaured_diff_y = np.power((trk_y.T - det_y), 2)

    sqaured_diff = sqaured_diff_x + sqaured_diff_y

    return np.sqrt(sqaured_diff.astype(np.float32))


def compute_diff_on_vec(vec):
    tf = np.isinf(vec)
    vec = np.array(vec)
    vec[tf] = 20 # value in meters --> very far
    vec = np.expand_dims(vec, axis=0)
    diff = vec.T - vec

    return diff
