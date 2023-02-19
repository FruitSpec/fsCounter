import matplotlib.pyplot as plt
from vision.tools.image_stitching import plot_2_imgs
import seaborn as sns

i = 7
for res in trk_outputs:
    crop_rgb = frame[max(res[1], 0):res[3], max(res[0], 0):res[2], :]
    crop_pc = point_cloud[max(res[1], 0):res[3], max(res[0], 0):res[2], :]
    crop_rgb_masked = crop_rgb.copy()

    # sns.kdeplot(crop_pc[:,:,2].flatten())
    # plt.vlines(0.52,0,60)
    # plt.show()
    flat_pc = crop_pc[:, :, 2].flatten()
    kernel = gaussian_kde(flat_pc[np.isfinite(flat_pc)])
    min_val, max_val = np.nanmin(crop_pc[:, :, 2]), min(np.nanmax(crop_pc[:, :, 2]), 1)
    x = np.linspace(min_val, max_val, 250)
    y = kernel(x)
    y = y / sum(y)
    picks = []
    bottoms = []
    for ind, val in enumerate(y[1:-1]):
        if y[ind] > val and y[ind + 2] > val and val > 0.0025:
            bottoms.append(ind + 1)
        if y[ind] < val and y[ind + 2] < val and val > 0.0025:
            picks.append(ind + 1)

    plt.scatter(x, y)
    for val in picks:
        plt.vlines(x[val], 0, y.max(), color="green")
    for val in bottoms:
        plt.vlines(x[val], 0, y.max(), color="red")
    plt.show()
    if len(picks):
        min_dist = x[picks[0]]
    else:
        min_dist = 0
    thresh_dist = np.min(np.where(np.all([y < 0.0025, min_dist < x], axis=0)))
    crop_rgb_masked[crop_pc[:, :, 2] > x[thresh_dist]] = 0
    plot_2_imgs(crop_rgb_masked, crop_rgb)
