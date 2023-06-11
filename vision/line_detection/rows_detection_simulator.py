import pandas as pd
import os
import cv2
from vision.line_detection.rows_detector import RowDetector
from vision.line_detection.retrieve_sensors_data import get_sensors_data
from tqdm import tqdm


# Get sensors data to csv file:
PATH_ROW = r'/home/lihi/FruitSpec/Data/customers/EinVered/2023_05_21/MAROR10D'
PATH_OUTPUT = r'/home/lihi/FruitSpec/Data/customers/EinVered/2023_05_21/MAROR10D/row_10/1'
output_dir = os.path.join(PATH_OUTPUT, 'rows_detection')
df = get_sensors_data(PATH_ROW, output_dir, GT = True, save=True)


CSV_PATH = os.path.join(output_dir, 'sensors_data.csv')
DEPTH_VIDEO_PATH = r'/home/lihi/FruitSpec/Data/customers/EinVered/2023_05_21/MAROR10D/row_10/1/DEPTH.mkv'
RGB_VIDEO_PATH = r'/home/lihi/FruitSpec/Data/customers/EinVered/2023_05_21/MAROR10D/row_10/1/ZED.mkv'
PATH_KML = r'/home/lihi/FruitSpec/Data/customers/EinVered/Blocks.kml'
EXPECTED_HEADING = None
#EXPECTED_HEADING = 100
POLYGON_NAME = 'POMELO00'

PATH_ROW = os.path.dirname(os.path.dirname(CSV_PATH))
output_dir = os.path.join(PATH_ROW, 'rows_detection')
output_name = "_".join(PATH_ROW.split('/')[-4:])

df = pd.read_csv(CSV_PATH)
print(f'Loaded {CSV_PATH}')

# init rows detector:
row_detector = RowDetector(expected_heading = EXPECTED_HEADING, path_kml = PATH_KML, placemark_name = POLYGON_NAME)

# load video:
cap_rgb = cv2.VideoCapture(RGB_VIDEO_PATH)
cap_depth = cv2.VideoCapture(DEPTH_VIDEO_PATH)
print(f'Loaded {DEPTH_VIDEO_PATH}')
print (f'Video_len:{int(cap_depth.get(cv2.CAP_PROP_FRAME_COUNT))}, df_len:{len(df)}')

# load frames:
for index, row in tqdm(df.iterrows(), total=df.shape[0]):

    # Get depth and RGB frames:
    ret1, depth_img = cap_depth.read()
    if not ret1:
        break
    ret2, rgb_img = cap_rgb.read()
    if not ret2:
        break

    #rotate images:
    depth_img = cv2.rotate(depth_img, cv2.ROTATE_90_CLOCKWISE)   #cv2.ROTATE_90_COUNTERCLOCKWISE
    rgb_img = cv2.rotate(rgb_img, cv2.ROTATE_90_CLOCKWISE)


    depth_img = depth_img[:, :, 0].copy()
    gt = row.GT if 'GT' in df.columns.values else None
    is_row, state_changed, df_results = row_detector.detect_row(depth_img=depth_img,
                                                                rgb_img=rgb_img,
                                                                angular_velocity_x = row.angular_velocity_x,
                                                                longitude = row.longitude, latitude = row.latitude,
                                                                imu_timestamp =row.timestamp,
                                                                gps_timestamp=row.timestamp_gnss,
                                                                ground_truth = gt)

    if cv2.waitKey(25) & 0xFF == ord('q'):    # Wait for 25 milliseconds and check if the user pressed 'q' to quit
        break


# save dataframe:
df_results.to_csv(os.path.join(output_dir, f'rows_data.csv'), index=False)
row_detector.generate_new_kml_file(output_dir=output_dir)

# plot sensors data:
config = f'EX3_Pred:Enter_depth_and_heading__Exit_ang_vel_DEPTH_THRESH {row_detector.DEPTH_THRESHOLD}, DEPTH_EMA {row_detector.DEPTH_EMA_ALPHA}, ANG_VEL_THRESH {row_detector.ANGULAR_VELOCITY_THRESHOLD}, ANG_VEL_EMA {row_detector.ANGULAR_VELOCITY_EMA_ALPHA}, EXPECTED_HEADING {row_detector.EXPECTED_HEADING}, HEADING_THRESH {row_detector.HEADING_THRESHOLD}_, self.MARGINS_THRESHOLD {row_detector.MARGINS_THRESHOLD}'

row_detector.plot_sensors(title = config + output_name,
             margins_threshold = row_detector.MARGINS_THRESHOLD,
             save_dir = output_dir)

print('Done!')
