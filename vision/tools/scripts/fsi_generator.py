import os
import sys
from PIL import Image
import numpy as np

from vision.tools.camera import generate_fsi_2

def run(input_folder, output_folder, gamma=4/5):

    folder_list = os.listdir(input_folder)
    output_folder_list = os.listdir(output_folder)

    for folder in folder_list:
        file_list = os.listdir(os.path.join(input_folder, folder))
        ids = []
        for file in file_list:
            if 'Stream' in file:
                temp = file.split('.')[0]
                id_ = temp.split('_')[-1]
                if id_ not in ids:
                    ids.append(id_)
        for id_ in ids:
            rgb_p = os.path.join(input_folder, folder, f'Stream0_{id_}.tiff')
            r_ch_p = os.path.join(input_folder, folder, f'Stream1_{id_}.tiff')
            g_ch_p = os.path.join(input_folder, folder, f'Stream2_{id_}.tiff')
            rgb = np.array(Image.open(rgb_p))
            r_ch = np.array(Image.open(r_ch_p))
            g_ch = np.array(Image.open(g_ch_p))

            fsi = generate_fsi_2(rgb, r_ch, g_ch, gamma)
            if not os.path.isdir(os.path.join(output_folder, folder)):
                os.mkdir(os.path.join(output_folder, folder))
            output_file_name = os.path.join(output_folder, folder, f"FSI_{id_}.jpg")
            Image.fromarray(fsi).save(output_file_name)


if __name__ == "__main__":

    input_folder = "/media/fruitspec-lab/262/JAI-test/300322"
    output_folder = "/home/fruitspec-lab/FruitSpec/Sandbox/Sliced_data/FSI4"
    run(input_folder, output_folder, 1)