import cv2
from vision.tools.manual_slicer import slice_to_trees
import numpy as np


data_file = "/home/lihi/FruitSpec/Data/customers/DEWAGD/190123/DWDBLE33/R35A/zed_rgb_slicer.json"
video_path = "/home/lihi/FruitSpec/Data/customers/DEWAGD/190123/DWDBLE33/R35A/zed_rgb.avi"
# slice_to_trees(data_file, file_path, output_path, resize_factor=3, h=2048, w=1536, on_fly=True)
df, df2 = slice_to_trees(data_file, video_path = video_path, resize_factor=3, output_path=None, h=1920, w=1080, on_fly=True)
FRAME_WIDTH = 1080
FRAME_HEIGHT = 1920

# COCO bounding box format is [top left x position, top left y position, width, height].
# load csv:
#df = pd.read_csv(r'/home/lihi/FruitSpec/Data/customers/JAIZED_CaraCara_151122/R_1/all_slices.csv', index_col = 0)
df.loc[df['start'] == -1, 'start'] = 0
df['x'] = df['start']
df['y'] = 0
df.loc[df['end'] == -1, 'end'] = FRAME_WIDTH
df['w'] = df['end'] - df['start']
df['h'] = FRAME_HEIGHT

# Identify the highest tree_id value
highest_tree_id = df['tree_id'].max()
# Remove rows based on the conditions
df = df[~((df['tree_id'] == highest_tree_id) & (df['x'] == 0) & (df['y'] == 0))]



cap = cv2.VideoCapture(video_path)

# Read and process each frame
paused = False  # Flag to determine if video display is paused

while cap.isOpened():
    if not paused:
        ret, frame = cap.read()

        if ret:
            frame_id = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1  # Get the current frame ID

            # Filter dataframe for the current frame ID
            frame_data = df[df['frame_id'] == frame_id]

            # Draw bounding boxes on the frame
            for _, row in frame_data.iterrows():
                x, y, w, h = int(row['x']), int(row['y']), int(row['w']), int(row['h'])
                line_thickness = 10
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 15)  # Draw the bounding box

            # Resize frame by a factor of 3 for display
            resized_frame = cv2.resize(frame, None, fx=1/2, fy=1/2)

            # Write frame number on the frame
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottom_left_corner = (10, 30)
            font_scale = 1
            font_color = (0, 255, 0)
            line_thickness = 2
            cv2.putText(resized_frame, f'Frame: {frame_id}', bottom_left_corner, font, font_scale, font_color,
                        line_thickness)

            # Display the resized frame with bounding boxes
            cv2.imshow('Frame', resized_frame)

        # Pause/resume video display on space key press
        key = cv2.waitKey(20)
        if key == ord(' '):
            paused = not paused
        elif key & 0xFF == ord('q'):
            break
    else:
        # Continue to check for space key press to resume video display
        key = cv2.waitKey(1)
        if key == ord(' '):
            paused = not paused

# Release the video capture object and close the windows
cap.release()
cv2.destroyAllWindows()


print ('done')
