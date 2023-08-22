import pandas as pd
import os
import cv2
from vision.line_detection.rows_detector import RowDetector
from tqdm import tqdm
import numpy as np
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import json
from vision.tools.utils_general import find_subdirs_with_file
from vision.line_detection.gps_distance import DistanceGPS


def read_imu_log(file_path, columns_names=['date', 'timestamp', 'angular_velocity_x', 'angular_velocity_y', 'angular_velocity_z', 'linear_acceleration_x', 'linear_acceleration_y', 'linear_acceleration_z']):
    try:
        with open(file_path, 'r') as file:
            log_data = file.readlines()

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the log file: {e}")
        return None

    table_data = []
    current_row = []

    for line in log_data:
        line = line.strip()  # Remove leading/trailing whitespace
        if line:
            #current_row += re.split(r'\s+|,', line)
            current_row += line.split()
        else:
            if current_row:
                table_data.append(current_row)
            current_row = []

    if current_row:
        table_data.append(current_row)

    if table_data:
        table = pd.DataFrame(table_data)
        table.iloc[:, 2:] = table.iloc[:, 2:].astype(float)
        table.columns = columns_names
        table = extract_time_from_timestamp(table)
        return table
    else:
        return None


def read_nav_file(file_path):
    try:
        df = pd.read_csv(file_path, header = None)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None

    expected_columns = ["timestamp", "latitude", "longitude", "plot"]
    if df.iloc[0].tolist() == expected_columns:
        df.columns = df.iloc[0]
        df = df[1:]

    else:
        df.columns = expected_columns

    return df



def extract_time_from_timestamp(df):
    # replace the letter 'O' with zero:
    df['timestamp'] = df['timestamp'].str.replace(r'O', '0', regex=True)
    # Convert timestamp to datetime:
    df['timestamp'] = pd.to_datetime(df['timestamp'], format='%H:%M:%S.%f')
    # Calculate time difference from the beginning of the video
    time_diff = df['timestamp'] - df['timestamp'].iloc[0]
    # Extract the minutes and seconds components from the time difference
    minutes = time_diff.dt.components['minutes'].astype(str).str.zfill(2)
    seconds = time_diff.dt.components['seconds'].astype(str).str.zfill(2)
    # Combine minutes and seconds into the desired format
    df['time_diff'] = minutes + ':' + seconds
    return df


def GT_to_df(GT, df):
    GT_intervals = [(pd.to_datetime(row.get('start_time'), format='%M:%S'), pd.to_datetime(row.get('end_time'), format='%M:%S')) for row in GT]
    # Create a list to store the flag values
    flags = []
    # Iterate over each row in the DataFrame
    for _, row in df.iterrows():
        # Convert the time_diff value to datetime
        frame_time = pd.to_datetime(row['time_diff'], format='%M:%S')
        # Check if the frame_time is within any of the GT_intervals
        is_in_GT = any(start <= frame_time <= end for start, end in GT_intervals)
        # Append the flag value to the list
        flags.append(1 if is_in_GT else 0)
    # Add the flags column to the DataFrame
    df['GT'] = flags
    return df


def plot_latitude_longitude(data, output_dir, save = True):
    # Create the plot
    plt.plot(data.longitude, data.latitude, 'b')
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('GNSS Data: Latitude vs Longitude')
    plt.grid(True)

    # Save the plot
    output_path = os.path.join(output_dir, 'GNSS_position.png')
    if save:
        plt.savefig(output_path)
        print (f'GNSS_position plot is saved to {output_path}')
    plt.show()

def plot_gnss_heading(data, output_dir, column = 'heading_gps', save = True):

    # Create the plot
    plt.figure(figsize=(20, 6))
    plt.plot(data['timestamp_gnss'], data[column])
    plt.xlabel('Time')
    plt.ylabel('Heading (degrees)')
    plt.title('GNSS Data: Heading by Time')
    plt.grid(True)

    # Format the x-axis as datetime
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
    plt.gcf().autofmt_xdate()

    output_path = os.path.join(output_dir, 'GNSS_heading.png')
    if save:
        plt.savefig(output_path)
        print(f'GNSS_heading plot is saved to {output_path}')
    plt.show()


def extract_gnss_data(df_imu, df_gps):
    # TODO - i loose index, if it's needed than should be fixed

    df_gps["timestamp_gnss"] = pd.to_datetime(df_gps["timestamp"], unit="ns").dt.time
    df_gps.drop('timestamp', axis='columns', inplace=True)

    df_imu["timestamp"] = pd.to_datetime(df_imu["timestamp"], unit="ns").dt.time

    # Extract the timestamp values from both DataFrames
    merged_timestamps = df_imu['timestamp'].values
    gps_timestamps = df_gps['timestamp_gnss'].values

    # Find the indices of the last matching timestamps in df_gps
    last_indices = gps_timestamps.searchsorted(merged_timestamps, side='right')
    if np.all(last_indices == -1):
        print("No matching GPS data.")
        return None

    # Filter df_gps using the last indices
    filtered_gps = df_gps.loc[last_indices]

    # Concatenate df_merged and filtered_gps
    merged_df = pd.concat([df_imu.reset_index(drop=True), filtered_gps.reset_index(drop=True)], axis=1)

    return merged_df



def get_sensors_data(PATH_ROW, output_dir, GT = True, save = False):
    # paths:
    path_log_imu = find_subdirs_with_file(PATH_ROW, 'imu.log', return_dirs=False, single_file=True)
    parent_dir = os.path.dirname(PATH_ROW)
    path_gps = find_subdirs_with_file(parent_dir, '.nav', return_dirs=False, single_file=True)

    gt_rows_path = os.path.join(output_dir, 'GT_lines.json')

    # load depth and imu data, and concat:
    df_imu = read_imu_log(path_log_imu)

    # load gps csv:
    df_gps = read_nav_file(path_gps)

    # merge with imu data:
    df_merged = extract_gnss_data(df_imu, df_gps)

    if GT:
        # Load rows ground_truth:
        with open(gt_rows_path, 'r') as file:
            GT_rows = json.load(file)
        df_merged = GT_to_df(GT_rows,df_merged)

    # Save df:
    if save:
        output_csv_path = os.path.join(output_dir, f'sensors_data.csv')
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        df_merged.to_csv(output_csv_path, index=False)
        print (f'Saved {output_csv_path}')

    return df_merged


if __name__ == "__main__":

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

    # init distance detector:
    dist_detector = DistanceGPS()

    # init rows detector:
    row_detector = RowDetector(expected_heading = EXPECTED_HEADING, path_kml = PATH_KML, placemark_name = POLYGON_NAME)

    # load video:
    cap_rgb = cv2.VideoCapture(RGB_VIDEO_PATH)
    cap_depth = cv2.VideoCapture(DEPTH_VIDEO_PATH)
    print(f'Loaded {DEPTH_VIDEO_PATH}')
    print (f'Video_len:{int(cap_depth.get(cv2.CAP_PROP_FRAME_COUNT))}, df_len:{len(df)}')

    initial_lon = df.longitude.iloc[0]
    initial_lat = df.latitude.iloc[0]
    accumulative_distance = 0

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
        ###############
        dist = dist_detector.get_distance(lon1=row.longitude, lat1=row.latitude)
        if dist !=0:
            accumulative_distance += dist
            print (f'lon: {row.longitude}, lat: {row.latitude}, distance: {dist}, accumulative_distance: {accumulative_distance}')

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

    row_detector.plot_sensors(title = config + output_name, save_dir = output_dir)

    print('Done!')
