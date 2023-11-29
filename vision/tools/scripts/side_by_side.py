import os
import cv2
import numpy as np
from tqdm import tqdm

from vision.tools.video_wrapper import video_wrapper

def moive_side_by_side(movie1_path, movie2_path, output_path, rot1=1, rot2=1, tot_frames=1000, fps=10, start_frame=0):
    width = 512 * 2
    height = 682
    output_video_name = os.path.join(output_path, 'result_video.avi')
    output_video = cv2.VideoWriter(output_video_name, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'),
                                   fps, (width, height))

    cam1 = video_wrapper(movie1_path, rotate=rot1)
    cam2 = video_wrapper(movie2_path, rotate=rot2)

    #plot, date, row_id = get_movie_meta(movie1_path)



    counter = 0
    progress_bar = tqdm(total=tot_frames, desc="Processing")
    while counter < start_frame + tot_frames:


        res1, frame1 = cam1.get_frame()
        res2, frame2 = cam2.get_frame()
        if not res1 or not res2:
            break
        if counter < start_frame:
            counter += 1
            continue
        progress_bar.update(1)

        frame1 = cv2.resize(frame1, (512, 682))
        frame2 = cv2.resize(frame2, (512, 682))
        canvas = np.hstack([frame1, frame2])

        output_video.write(canvas)

        counter += 1

    print('Done')

    output_video.release()
    cam1.close()
    cam2.close()


def get_movie_meta(movie_path):

    splited = movie_path.split('/')
    row_id = splited[-3]
    date = splited[-4]
    plot = splited[-5]

    return plot, date, row_id

if __name__ == "__main__":
    row = r"C:\Users\Matan\Documents\SA_apples\row_2222\1"
    movie1_path = os.path.join(row, "Result_FSI.mkv")
    movie2_path = os.path.join(row, "FSI_CLAHE.mkv")
    output_path = row
    moive_side_by_side(movie1_path, movie2_path, output_path, rot1=1, rot2=1, tot_frames=300, fps=10, start_frame=0)