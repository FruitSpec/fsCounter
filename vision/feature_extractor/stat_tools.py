import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.metrics import pairwise_distances
from scipy.sparse import csr_matrix
from scipy.stats import gaussian_kde

def quantile_trim(data_arr, trim_vals=(0.025, 0.975), keep_size=True):
    """
    :param data_arr: data to trim
    :param trim_vals:  top and buttom quantiles (everything not in the interval will be dropped)
    :return: trimmed array
    """
    if len(data_arr) < 5:
        return data_arr
    data_arr_q = np.quantile(data_arr, trim_vals)
    if len(data_arr_q) < 2:
        return data_arr
    if isinstance(data_arr, list):
        data_arr = np.array(data_arr)
    if not keep_size:
        data_arr = data_arr[np.all([data_arr >= data_arr_q[0], data_arr <= data_arr_q[1]], axis=0)]
    else:
        data_arr[data_arr <= data_arr_q[0]] = np.nan
        data_arr[data_arr >= data_arr_q[1]] = np.nan
    return data_arr


def iqr_trim(arr, keep_size=True, n_std=3):
    """
    trims data based on the iqr method
    :param arr: array to work on
    :param keep_size: flag to keep size or not (if false will drop trimmed data, else will replace with nan)
    :param n_std: number of stds to keep
    :return: trimmed array
    """
    qauntiles = np.nanquantile(arr, (0.25, 0.75))
    iqr_val = qauntiles[1] - qauntiles[0]
    valid_range = qauntiles + np.array([-n_std*iqr_val, n_std*iqr_val])
    if keep_size:
        arr[arr < valid_range[0]] = np.nan
        arr[arr > valid_range[1]] = np.nan
    else:
        arr = arr[np.all([arr > valid_range[0], arr < valid_range[1]], axis=0)]
    return arr


def iqr_max(arr, n_std=2, max_quantile=0.9):
    # TODO when arrr is empty return of non na values return somthing
    qauntiles = np.nanquantile(arr, (0.25, 0.75, max_quantile))
    if isinstance(qauntiles, float):
        return 0
    t_quantile = qauntiles[1]
    iqr_val = t_quantile - qauntiles[0]
    return min(t_quantile + n_std*iqr_val, qauntiles[2])


def smooth_data_np_average(arr, span=2, trim=True, keep_nans = False):
    """
    runs a moving average on the array to smooth the data
    :param arr: array
    :param span: number of extra samples to take from each side
    :return: a smothed array
    """
    if len(arr.shape) == 1:
        len_arr = len(arr)
        if trim:
            arr = quantile_trim(arr)
        if len(arr) < span*2 +1:
            return np.nan
        out_arr = np.array([np.nanmean(arr[max(val - span, 0):min(val + span + 1, len_arr)]) for val in range(len_arr)])
        if keep_nans:
            out_arr[np.isnan(arr)] = np.nan
        return out_arr
    y_shape, x_shape = arr.shape
    if trim:
        arr = quantile_trim(arr.flatten()).reshape((y_shape, x_shape))
    for x in range(x_shape):
        for y in range(y_shape):
            arr[y, x] = np.nanmean(arr[max(y - span, 0):min(y + span + 1, y_shape),
                                  max(x - span, 0):min(x + span + 1, x_shape)])
    return arr


def quantile_trim_mean(data_arr, trim_vals=(0.025, 0.975), keep_size=True):
    """
    :param data_arr: data to trim
    :param trim_vals:  top and buttom quantiles (everything not in the interval will be dropped)
    :param keep_size: flag to keep size or not (if false will drop trimmed data, else will replace with nan)
    :return: the mean of the trimmed array
    """
    return np.nanmean(quantile_trim(data_arr, trim_vals, keep_size))


def compute_density_mst(centers):
    """
    :param centers: centers of points
    :return: mst, distances
    """
    try:
        dist = pairwise_distances(centers)
    except:
        print("FD")
    mst = minimum_spanning_tree(csr_matrix(dist))
    mst_arr = mst.toarray()
    distances = mst_arr[mst_arr != 0]
    distances = distances[~np.isnan(distances)]
    return mst, distances


def get_mode_kde(data_arr, step=0.05):
    """
    :param data_arr: data_array to use
    :param step: step value for arange
    :return: mode of the distribution (not the array itself)
    """
    if len(data_arr) < 5:
        return np.nan
    kde = gaussian_kde(data_arr)
    min_val, max_val = np.min(data_arr), np.max(data_arr)
    x_values = np.arange(min_val, max_val, step=step)
    pdf_estimated_values = kde(x_values)
    return x_values[np.argmax(pdf_estimated_values)]