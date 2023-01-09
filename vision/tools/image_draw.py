import cv2
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import pandas as pd


def draw_tree_slicing(tree_folder, output_folder, slicer_results):
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)
    frames = [int(file.split(".")[0].split("_")[-1]) for file in os.listdir(tree_folder)
                  if "jpg" in file and "FSI" in file]
    for frame in frames:
        fsi = f"channel_FSI_frame_{frame}.jpg"
        rgb_zed = f"frame_{frame}.jpg"
        rgb_jai = f"channel_RGB_frame_{frame}.jpg"
        for picture_name in [fsi, rgb_zed, rgb_jai]:
            img = cv2.imread(os.path.join(tree_folder, picture_name))
            max_y = img.shape[0] - 1
            img = cv2.line(img, (slicer_results[frame][0], 0), (slicer_results[frame][0], max_y), color=(255, 0, 0),
                           thickness=3)
            img = cv2.line(img, (slicer_results[frame][1], 0), (slicer_results[frame][1], max_y), color=(255, 0, 0),
                           thickness=3)
            cv2.imwrite(os.path.join(output_folder, picture_name), img)


if __name__ == '__main__':
    trees_folder = "/media/fruitspec-lab/easystore/track_detect_analysis"
    for tree in tqdm(os.listdir(trees_folder)):
        tree_path = os.path.join(trees_folder, tree)
        if (not os.path.isdir(tree_path)) or tree == "clean":
            continue
        slicer_res = pd.read_csv(os.path.join(tree_path, "slices.csv"))[["frame_id", "start", "end"]]
        slicer_res.replace(-1, 0, inplace=True)
        if tree.startswith("R11") and (not tree.endswith("T23")):
            slicer_res["start"] = slicer_res["start"]/3*4
            slicer_res["end"] = slicer_res["end"] / 3 * 4
        slicer_results = {row["frame_id"]: (int(row["start"]), int(row["end"])) for i, row in slicer_res.iterrows()}
        draw_tree_slicing(tree_path, tree_path, slicer_results)