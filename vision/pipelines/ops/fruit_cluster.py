import os
import numpy as np
import collections

class Cluster():
    def __init__(self, cluster_id, track_ids):
        self.cluster_id = cluster_id
        self.track_ids = track_ids
        self.cluster_center = (-1, -1)
        self.lost = 0

class FruitCluster():
    def __init__(self,
                 max_single_fruit_dist=150,
                 x_dist_threshold=70,
                 y_dist_threshold=50,
                 range_diff_threshold=0.05,
                 max_losses=5):

        self.clusters = []
        self.cluster_id = 0
        self.max_single_fruit_dist = max_single_fruit_dist
        self.range_diff_threshold = range_diff_threshold
        self.x_dist_threshold = x_dist_threshold
        self.y_dist_threshold = y_dist_threshold
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

        neighbors = self.get_neighbors(trk_results, ranges)

        c_ids_to_t_ids, t_ids_to_c_ids = self.find_matched_clusters(neighbors)

        trk_to_cluster, live_clusters = self.update_new_clusters(t_ids_to_c_ids, neighbors)

        not_related_t_ids = []
        for t_id, c_ids in t_ids_to_c_ids.items():
            if len(c_ids) == 1:

                trk_to_cluster, live_clusters = self.update_clusters(neighbors, t_id, c_ids[0], trk_to_cluster, live_clusters)

            elif len(c_ids) > 1:
                counts = []
                track_ids_ = list(neighbors[t_id])
                for c_id in c_ids:
                    count = 0
                    cluster = self.clusters[c_id]
                    for track_id in track_ids_:
                        if track_id in cluster.track_ids:
                            count += 1
                    counts.append(count)

                best_cluster_id = np.argmax(counts)

                # neighbors centers
                #t_ids_center_x, t_ids_center_y = get_center(neighbors, trk_results, t_id)

                # cluster center
                #c_center_x, c_center_y = get_clusters_center(trk_results, self.clusters, c_ids)

                #best_cluster_id = find_best_neighbour(c_center_x, c_center_y, t_ids_center_x, t_ids_center_y)
                c_id = c_ids[best_cluster_id]
                # for index_, id_ in enumerate(c_ids):
                #     if index_ == best_cluster_id:
                #         continue
                #     not_related_t_ids.append(id_)

                trk_to_cluster, live_clusters = self.update_clusters(neighbors, t_id, c_id, trk_to_cluster,
                                                                     live_clusters)

        #         if t_id in not_related_t_ids:
        #             not_related_t_ids.remove(t_id)
        #
        # if len(not_related_t_ids) > 0:
        #     trk_to_cluster, live_clusters = self.add_not_related(neighbors,
        #                                                          not_related_t_ids,
        #                                                          trk_to_cluster,
        #                                                          live_clusters)

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
        output = trk_results.copy()
        cluster_ids = []
        for trk in output:
            cluster_id = trk_to_cluster[trk[6]]
            trk.append(cluster_id)
            cluster_ids.append(cluster_id)

        return output, cluster_ids


    def update_clusters(self, neighbors, t_id, c_id, trk_to_cluster, live_clusters):

        for t_n in neighbors[t_id]:
            trk_to_cluster[t_n] = self.clusters[c_id].cluster_id
            if t_n in self.clusters[c_id].track_ids:
                continue
            self.clusters[c_id].track_ids.append(t_n)

        if self.clusters[c_id].cluster_id not in live_clusters:
            live_clusters.append(self.clusters[c_id].cluster_id)

        return trk_to_cluster, live_clusters


    def get_neighbors(self, trk_results, ranges):
        trk_results = np.array(trk_results)
        dist = compute_dist_on_vec(trk_results, trk_results)

        range_x, range_y = compute_xy_ranges(trk_results)
        diff = compute_diff_on_vec(ranges)

        t_ids = np.array([trk[6] for trk in trk_results])
        neighbors = []
        # for t_dist, t_diff in zip(dist, diff):
        #    dist_bool = t_dist < self.max_single_fruit_dist
        for x_dist, y_dist, t_diff in zip(range_x, range_y, diff):
            x_dist_bool = x_dist <= self.x_dist_threshold
            y_dist_bool = y_dist <= self.y_dist_threshold
            diff_bool = abs(t_diff) < self.range_diff_threshold
            t_id_neighbors = t_ids[x_dist_bool & y_dist_bool & diff_bool]

            is_in_neighbors = False
            for neighbor in neighbors:
                tf = [False] * len(t_id_neighbors)
                for id_, t_id_neighbor in enumerate(t_id_neighbors):
                    if t_id_neighbor in neighbor:
                        tf[id_] = True
                if np.sum(tf) == len(t_id_neighbors):
                    is_in_neighbors = True
                    break

            if not is_in_neighbors:
                neighbors.append(t_id_neighbors)

            final_neighbors = []
            for n_id, neighbor in enumerate(neighbors):
                is_in_neighbors = False
                for id_, neighbor_compare in enumerate(neighbors):
                    if id_ == n_id:
                        continue
                    tf = [False] * len(neighbor)
                    for t_id, t in enumerate(neighbor):
                        if t in neighbor_compare:
                            tf[t_id] = True
                    if np.sum(tf) == len(neighbor):
                        is_in_neighbors = True

                if not is_in_neighbors:
                    final_neighbors.append(neighbor)



        return final_neighbors

    def find_matched_clusters(self, neighbors):
        c_ids_to_t_ids = {}
        t_ids_to_c_ids = {}
        for n_id, t_id_neighbors in enumerate(neighbors):
            for c_id, c in enumerate(self.clusters):
                for n in c.track_ids:
                    if n in t_id_neighbors:
                        if c in list(c_ids_to_t_ids.keys()):
                            c_ids_to_t_ids[c_id].append(n_id)
                        else:
                            c_ids_to_t_ids[c_id] = [n_id]

                        break

        for c_id, c in enumerate(self.clusters):
            for n_id, t_id_neighbors in enumerate(neighbors):
                for n in t_id_neighbors:
                    if n in c.track_ids:
                        if n_id in list(t_ids_to_c_ids.keys()):
                            if c_id not in t_ids_to_c_ids[n_id]:
                                t_ids_to_c_ids[n_id].append(c_id)
                        else:
                            t_ids_to_c_ids[n_id] = [c_id]

                        break


        return c_ids_to_t_ids, t_ids_to_c_ids

    def update_new_clusters(self, t_ids_to_c_ids, neighbors):
        trk_to_cluster = {}
        live_clusters = []

        neighbors_with_cluster = list(t_ids_to_c_ids.keys())
        for n_id, _ in enumerate(neighbors):
            if n_id in neighbors_with_cluster:
                continue

            self.clusters.append(Cluster(self.cluster_id, list(neighbors[n_id])))
            for t_n in list(neighbors[n_id]):
                trk_to_cluster[t_n] = self.cluster_id
            live_clusters.append(self.cluster_id)

            self.cluster_id += 1

        return trk_to_cluster, live_clusters

    def add_not_related(self, neighbors, not_related, trk_to_cluster, live_clusters):

        for t_id in not_related:

            self.clusters.append(Cluster(self.cluster_id, list(neighbors[t_id])))
            for t_n in list(neighbors[t_id]):
                trk_to_cluster[t_n] = self.cluster_id
            live_clusters.append(self.cluster_id)

            self.cluster_id += 1

        return trk_to_cluster, live_clusters

def get_center(neighbors, trk_results, t_id):
    neighbors_center_x = []
    neighbors_center_y = []
    arr = np.array(trk_results)
    trck_ids = arr[:, 6]
    for trck_id in neighbors[t_id]:
        index_ = np.where(trck_ids == trck_id)[0][0]
        bbox = arr[int(index_), :4]
        neighbors_center_x.append((bbox[0] + bbox[2]) // 2)
        neighbors_center_y.append((bbox[1] + bbox[3]) // 2)

    return np.mean(neighbors_center_x), np.mean(neighbors_center_y)
def get_clusters_center(trk_results, clusters, c_ids):
    centers_x = []
    centers_y = []

    for c in c_ids:
        c_center_x, c_center_y = get_cluster_center(trk_results, clusters, c)
        centers_x.append(c_center_x)
        centers_y.append(c_center_y)

    return centers_x, centers_y

def get_cluster_center(trk_results, clusters, c_id):
    arr = np.array(trk_results)
    frame_tracks = list(arr[:, 6])
    cluster_center_x = []
    cluster_center_y = []
    for clustr_track in clusters[c_id].track_ids:
        if clustr_track in frame_tracks:
            trck_id = frame_tracks.index(clustr_track)
            bbox = arr[trck_id, :4]
            cluster_center_x.append((bbox[0] + bbox[2]) // 2)
            cluster_center_y.append((bbox[1] + bbox[3]) // 2)

    c_center_x = np.mean(cluster_center_x)
    c_center_y = np.mean(cluster_center_y)

    return c_center_x, c_center_y

def find_best_neighbour(c_center_x, c_center_y, t_id_center_x, t_id_center_y):
    x_delta = np.array(c_center_x) - t_id_center_x
    y_delta = np.array(c_center_y) - t_id_center_y

    delta = np.sqrt(np.power(x_delta, 2) + np.power(y_delta, 2))

    best_neighbors_id = np.argmin(delta)

    return best_neighbors_id


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


def compute_xy_ranges(data):
    dets = data[:, :4]
    vec_x = (dets[:, 2] + dets[:, 0]) // 2
    vec_y = (dets[:, 3] + dets[:, 1]) // 2

    diff_x = np.abs(compute_diff_on_vec(vec_x))
    diff_y = np.abs(compute_diff_on_vec(vec_y))

    return diff_x, diff_y
