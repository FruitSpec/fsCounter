import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d
from tqdm import tqdm

from vision.tools.camera import find_gl_by_percentile



def slice_frame(depth, window_thrs=0.7, signal_thrs=0.5, neighbours_thrs=250):

    s, e = find_tree_height_limits(depth)

    cropped_depth = depth[s:e, :].copy()
    vec = find_local_minimum(cropped_depth, signal_thrs=signal_thrs)
    vec = remove_windows(depth, vec, s, e, window_thrs=window_thrs)
    if len(vec) > 1:
        vec = reduce_neighbours(depth, vec, s, e, neighbours_thrs=neighbours_thrs)


    return vec


def find_local_minimum(cropped_depth, signal_thrs=0.4, gaussian_kernel=5):
    mean_vec = np.mean(cropped_depth, axis=0)
    mean_vec = gaussian_filter1d(mean_vec, gaussian_kernel)

    k = [1, 0, -1]
    first_der_vec = np.convolve(mean_vec, k, mode='same') / 3
    second_der_vec = np.convolve(first_der_vec, k, mode='same') / 3

    # first derivative suspects - both min and max points
    tf_f = np.abs(first_der_vec) <= 0.03

    # second derivative suspects - keep only local min
    tf_s = second_der_vec > 0

    # keep only local min below threshold - far
    norm_vec = mean_vec / mean_vec.max()
    tf_n = norm_vec < signal_thrs

    vec = tf_f & tf_s & tf_n

    return reduce_to_center(vec)


def remove_windows(depth, vec, start, end, signal_thrs=10, window_thrs=0.7):
    final = []
    for index_ in vec:
        mid = (start + end) // 2

        # seperate the signal to two halfs
        uppper_signal_vec = depth[start:mid, index_].copy()
        lower_signal_vec = depth[mid: end, index_].copy()

        # how many below threshold - no signal
        upper = np.sum(uppper_signal_vec < signal_thrs) / (mid - start)
        lower = np.sum(lower_signal_vec < signal_thrs) / (end - mid)

        if upper > window_thrs and lower > window_thrs:
            # valid
            final.append(index_)
        elif upper > window_thrs and lower < window_thrs:
            # upper window
            final.append(index_)
        elif upper < window_thrs and lower > window_thrs:
            # lower window
            continue
        #else:  # gap - valid
        #    final.append(index_)

    return final

def reduce_neighbours(depth, vec, start, end, neighbours_thrs=250):
    last = vec[0]
    neighbours = [last]
    reduced = []
    for i in vec[1:]:
        if i - neighbours[-1] < neighbours_thrs:
            neighbours.append(i)
        else:
            score = np.array([])
            for n in neighbours:
                score = np.append(score, np.sum(depth[start:end, n]))
            j = np.argmin(score)
            reduced.append(neighbours[j])

            neighbours = [i]
            if i == vec[-1]:  # last
                reduced.append(i)

    # for loop finished before taking care of neighbours list
    if len(neighbours) > 1:
        score = np.array([])
        for n in neighbours:
            score = np.append(score, np.sum(depth[start:end, n]))
        j = np.argmin(score)
        reduced.append(neighbours[j])

    if len(reduced) > 2:
        reduced = [reduced[0], reduced[-1]]

    return reduced


def reduce_to_center(vec):
    centers = []
    args = np.argwhere(vec == 1)
    if len(args) == 0:
        return centers
    temp = [args[0][0]]
    for i in range(1, len(args)):
        if args[i][0] - args[i - 1][0] <= 1:
            temp.append(args[i][0])
        else:
            centers.append(int(np.mean(temp)))
            temp = [args[i][0]]

    centers.append(int(np.mean(temp)))

    return centers



def find_tree_height_limits(depth, fov_slice=7, filter_width=11, noise_thrs=20, median_kernel=15, min_start=750, min_end=1350 ):
    depth = cv2.medianBlur(depth, median_kernel)

    w = depth.shape[1]
    slices = np.arange(0, w, w // fov_slice).astype(np.int16)
    slices = np.append(slices, w)

    s = 0
    vecs = []
    for i in slices[1:]:
        v = np.mean(depth[:, s:i], axis=1)
        v = gaussian_filter1d(v, filter_width)
        vecs.append(v)
        s = i

    d = []
    v_l = vecs[0]
    for v in vecs[1:]:
        d.append(np.abs(v - v_l))
        v_l = v.copy()

    d = np.array(d)
    d[d <= noise_thrs] = 0

    starts = np.median(first_nonzero(d, axis=1)).astype(np.uint16)
    ends = np.median(last_nonzero(d, axis=1)).astype(np.uint16)

    starts = min(min_start, starts)
    ends = max(min_end, ends)

    return starts, ends



# def slice_frame(depth, routh_thrs=4000, grad_kernel=3, number_of_rouths=30, depth_thrs=70, get_best=4, delta_thrs=500):
#
#     pos_grad, neg_grad = get_grad(depth, k_size=grad_kernel)
#
#     # mask using depth - leave only remote areas
#     neg_grad[depth > depth_thrs] = 0
#     pos_grad[depth > depth_thrs] = 0
#
#     n_xs, n_ys, n_mes = extract_rouths(neg_grad, routh_thrs, number_of_rouths, get_best)
#     p_xs, p_ys, p_mes = extract_rouths(pos_grad, routh_thrs, number_of_rouths, get_best)
#
#     gaps = filter_by_delta(n_xs, n_ys, n_mes, p_xs, p_ys, p_mes, delta_thrs)
#
#     return gaps


def extract_rouths(grad, routh_thrs, number_of_rouths, get_best=4):
    first, last = get_search_limits(grad)
    args = get_initial_search_position(grad, first, last)
    #args = refine_args(args)
    if len(args) < number_of_rouths:
        number_of_rouths = len(args)
    xs, ys, mes, nzs = get_rouths_candidates(grad, args[:number_of_rouths], first, last)
    xs, ys, mes = finalize(xs, ys, mes, nzs, routh_thrs, get_best)

    return xs, ys, mes


def finalize(xs, ys, mes, nzs, routh_thrs, get_best=4, similar_thrs=20, non_zero_thrs=0.6):
    xs, ys, mes = remove_similar(xs, ys, mes, thrs=similar_thrs)
    xs, ys, mes = remove_zeros(xs, ys, mes, nzs, non_zero_thrs)
    sorted_args = np.argsort(np.array(mes))
    if len(sorted_args) > get_best:
        sorted_args = sorted_args[-get_best:]

    f_xs = []
    f_ys = []
    f_mes = []
    for arg in sorted_args:
        if mes[arg] > routh_thrs:
            f_xs.append(xs[arg])
            f_ys.append(ys[arg])
            f_mes.append(mes[arg])

    return f_xs, f_ys, f_mes

def remove_zeros(xs, ys, mes, nzs, non_zero_thrs=0.66):
    f_xs = []
    f_ys = []
    f_mes = []
    for i in range(len(ys)):
        if (nzs[i] / len(ys[i])) > non_zero_thrs:
            f_xs.append(xs[i])
            f_ys.append(ys[i])
            f_mes.append(mes[i])

    return f_xs, f_ys, f_mes


def remove_similar(xs, ys, mes, thrs=20):
    xs = np.array(xs)
    ys = np.array(ys)

    tested = []
    remove = []
    for i in range(xs.shape[0]):
        for j in range(xs.shape[0]):
            if i == j:
                continue
            delta = np.mean(np.abs(xs[j, :] - xs[i, :]))
            if delta <= thrs:
                if j in tested:
                    continue
                if j not in remove:
                    remove.append(j)
                    if i not in tested:
                        tested.append(i)

    r_xs = []
    r_ys = []
    r_mes = []
    for i in range(xs.shape[0]):
        if i not in remove:
            r_xs.append(list(xs[i, :]))
            r_ys.append(list(ys[i, :]))
            r_mes.append(mes[i])

    return r_xs, r_ys, r_mes


def get_routh(grad, start, first_line, last_line):
    f = first_line
    l = last_line
    x1, y1, e1, nz1 = find_best_routh(grad, [(f + l) // 2, start], f)
    x2, y2, e2, nz2 = find_best_routh(grad, [(f + l) // 2, start], l)

    x1.reverse()
    y1.reverse()
    x = x1 + x2
    y = y1 + y2
    energy = e1 + e2
    nz = nz1 + nz2

    return x, y, np.mean(energy), nz


def get_rouths_candidates(grad, args, first_line, last_line):
    xs = []
    ys = []
    mes = []
    nzs = []
    for arg in args:
        x, y, me, nz = get_routh(grad, int(arg), first_line, last_line)
        xs.append(x)
        ys.append(y)
        mes.append(me)
        nzs.append(nz)

    return xs, ys, mes, nzs

def remove_outliers(depth, upper, lower):
    u, l = find_gl_by_percentile(depth, upper, lower)
    depth = np.clip(depth, l, u)

    return depth

def get_grad(depth, direction='x', k_size=3, upper=0.98, lower=0.02):

    if direction == 'x':
        grad = cv2.Sobel(depth, cv2.CV_16S, 1, 0, ksize=k_size, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)
    else:
        grad = cv2.Sobel(depth, cv2.CV_16S, 0, 1, ksize=k_size, scale=1, delta=0, borderType=cv2.BORDER_DEFAULT)

    grad = remove_outliers(grad, upper, lower)

    pos_grad = grad.copy()
    pos_grad[grad < 0] = 0

    neg_grad = grad.copy()
    neg_grad[grad > 0] = 0

    return pos_grad, neg_grad


def refine_args(args, max_delta=20):
    last = 0
    i = 0
    res = []
    c = []
    for a in args:
        if np.abs(a - last) > max_delta:
            if len(c) > 0:
                res.append(np.mean(c))
            i += 1
            c = []
        else:
            c.append(a)
        last = a

    return res


def get_initial_search_position(grad, first_line, last_line):
    #v = grad[(first_line + last_line) // 2, :]
    #args = np.argsort(v)
    l = grad.shape[1]
    args = np.arange(15, (l - 15), (l - 30) / 30)

    return args

def get_search_limits(grad, direction='x', noise_lvl = 0.25):
    if direction == 'x':
        axis = 1
    else:
        axis = 0

    energy = np.abs(np.sum(grad, axis=axis))

    # remove noise
    noise_threshold = energy.max() * noise_lvl
    energy[energy < noise_threshold] = 0

    last = last_nonzero(energy, axis=0)
    first = first_nonzero(energy, axis=0)

    return first, last

def last_nonzero(arr, axis, invalid_val=-1):
    mask = arr != 0
    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    return np.where(mask.any(axis=axis), val, invalid_val)

def first_nonzero(arr, axis, invalid_val=-1):
    mask = arr != 0
    return np.where(mask.any(axis=axis), mask.argmax(axis=axis), invalid_val)


def find_best_routh(grad, start_position, end_line):

    weights = np.array([0.5, 0.7, 1, 0.7, 0.5])
    x_position = start_position[1]
    xs = []
    ys = []
    e = []
    non_zero = 0
    step = 1 if end_line > start_position[0] else -1
    for i in range(start_position[0], end_line, step):
        signal_min_arg = max(x_position - 2, 0)
        signal_max_arg = min(x_position + 3, grad.shape[1])
        signal_vec = np.abs(grad[i, signal_min_arg:signal_max_arg])
        if len(signal_vec) == 5:
            w_signal_vec = signal_vec #* weights
        else:
            w_signal_vec = signal_vec

        if np.sum(w_signal_vec) > 0:  # not all zeros
            vec_arg = np.argmax(w_signal_vec)
            x_position = signal_min_arg + vec_arg
            non_zero += 1
        xs.append(x_position)
        ys.append(i)
        e.append(np.abs(grad[i, x_position]))

    return xs, ys, np.sum(e), non_zero


def filter_by_delta(n_xs, n_ys, n_mes, p_xs, p_ys, p_mes, delta_thrs=500, abs=False):

    n_xs = np.array(n_xs)
    n_ys = np.array(n_ys)
    p_xs = np.array(p_xs)
    p_ys = np.array(p_ys)
    common = min(n_ys.shape[1], p_ys.shape[1])

    gaps = {}
    count = 0
    for i in range(n_xs.shape[0]):
        for j in range(p_xs.shape[0]):
            if abs:
                delta = np.mean(np.abs(p_xs[j, -common:] - n_xs[i, -common:]))
            else:
                delta = np.mean(p_xs[j, -common:] - n_xs[i, -common:])
            if delta < 0:
                continue
            elif delta <= delta_thrs:
                gaps[count] = {'n_xs':  n_xs[i, :],
                               'n_ys': n_ys[i, :],
                               'n_mes': n_mes[i],
                               'p_xs': p_xs[j, :],
                               'p_ys': p_ys[j, :],
                               'p_mes': p_mes[j],
                               'delta': delta}
                count += 1

    return gaps


def print_lines(frame, depth, lines):
    out = frame.copy()
    s, e = find_tree_height_limits(depth)
    for line in lines:
        out = cv2.line(out, (line, s), (line, e), (255, 0, 255), 2)

    return out



if __name__ == "__main__":
    f_ids = np.arange(136, 278, 1)
    #f_ids = [164]

    data = {}
    for f_id in tqdm(f_ids):
        fp = f"/home/yotam/FruitSpec/Sandbox/slicer_test/caracara_R2_3011/sliced3/depth/depth_frame_{f_id}.jpg"
        depth = cv2.imread(fp)
        depth = cv2.cvtColor(depth, cv2.COLOR_BGR2GRAY)
        fp = f"/home/yotam/FruitSpec/Sandbox/slicer_test/caracara_R2_3011/sliced3/frames/frame_{f_id}.jpg"
        frame = cv2.imread(fp)
        #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #s, e = find_tree_height_limits(depth)
        gaps = slice_frame(depth, window_thrs=0.4)

        output = print_lines(frame, depth, gaps)
        fp = f"/home/yotam/FruitSpec/Sandbox/slicer_test/caracara_R2_3011/sliced3/slice/frame_slice_{f_id}.jpg"
        cv2.imwrite(fp, output)

        data[f_id] = gaps




