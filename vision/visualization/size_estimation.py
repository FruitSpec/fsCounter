import os
import cv2
import pandas as pd


fp = r"C:\Users\Matan\Documents\Projects\Data\Caliber_size_variation"
file_list = os.listdir(fp)
for file in file_list:
    if 'xlsx' in file.split('.')[-1] and '$' not in file:
        xl = file
    elif 'mp4' in file.split('.')[-1]:
        clip = file

xl_file_path = os.path.join(fp, xl)
clip_file_path = os.path.join(fp, clip)
df = pd.read_excel(xl_file_path, sheet_name='filtered')
frames_list = list(df['frame'])

def split_clip_to_frames(fp, clip_file_path, frames_list):

    cap = cv2

    # Check if camera opened successfully
    if (cap.isOpened() == False):
        print("Error opening video stream or file")

    # Read until video is completed
    output = os.path.join(fp, 'frames')
    if not os.path.exists(output):
        os.mkdir(output)
    f_id = 0
    ids = []
    while (cap.isOpened()):

        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:

            if f_id in frames_list:
                frame_output_path = os.path.join(output, f"frame_{f_id}.jpg")
                cv2.imwrite(frame_output_path, frame)

            f_id += 1

        # Break the loop
        else:
            break

    # When everything done, release the video capture object
    cap.release()

    # Closes all the frames
    cv2.destroyAllWindows()
    print('Done')



