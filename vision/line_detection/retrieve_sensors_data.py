import logging

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np
import os
import re
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from geographiclib.geodesic import Geodesic
from tqdm import tqdm
import datetime
import json
from vision.tools.utils_general import find_subdirs_with_file, variable_exists


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

def plot_sensors(df, title, depth_threshold = 0.5, angular_velocity_threshold = 10, expected_heading = None, lower_heading_bound= None, upper_heading_bound= None, save_dir=None,  margins_threshold =None): #,

    # if there is gps data, make 4 subplots, else 2.
    if "heading_360" in df.columns.values:
        n_subplots = 4
        plt.figure(figsize=(55, 35))

    else:
        n_subplots = 2
        plt.figure(figsize=(55, 18))

    # Plot:
      # Adjust the width and height as desired
    sns.set(font_scale=2)

    n_subplots = 6
    _subplot_(df, n_subplots = n_subplots, i_subplot = 1, column_name1="score", column_name2="depth_ema",
              thresh1=depth_threshold, thresh2 = None, thresh3 = None, title ='Depth score')

    _subplot_(df, n_subplots = n_subplots, i_subplot = 2, column_name1="angular_velocity_x", column_name2="ang_vel_ema",
              thresh1=angular_velocity_threshold, thresh2 = -angular_velocity_threshold, thresh3 = None, title ='angular_velocity_x (deg/sec)')

    _subplot_(df, n_subplots = n_subplots, i_subplot = 3, column_name1="heading_180", column_name2=None,
              thresh1=lower_heading_bound, thresh2 = upper_heading_bound, thresh3 = expected_heading, title ='heading_180')

    _subplot_(df, n_subplots = n_subplots, i_subplot = 4, column_name1="within_inner_polygon", column_name2=None,
              thresh1=None, thresh2 = None, thresh3 = None, title =f"within_inner_polygon")

    _subplot_(df, n_subplots = n_subplots, i_subplot = 5, column_name1='row_state', column_name2=None,
              thresh1=None, thresh2 = None, thresh3 = None, title =f"'Row_state. 0: 'not in row', 1: 'starting row', 2: 'middle of row', 3: 'end of row'")

    _subplot_(df, n_subplots = n_subplots, i_subplot = 6, column_name1="pred", column_name2=None,
              thresh1=None, thresh2 = None, thresh3 = None, title =f"Prediction. 0: 'not in row', 1: 'starting row'")



    plt.suptitle(title)
    plt.tight_layout()

    if save_dir:
        output_path = os.path.join(save_dir, f"plot_{title}.png")
        plt.savefig(output_path)
        plt.close()
        print (f'saved plot to {output_path}')

    plt.show()

def _subplot_(df, n_subplots, i_subplot, title, column_name1, column_name2 = None, thresh1= None, thresh2= None, thresh3= None):

    plt.subplot(n_subplots, 1, i_subplot)

    # draw the ground truth:
    if 'GT' in df.columns:
        plt.fill_between(df.index, df[column_name1].min(), df[column_name1].max(), where=df['GT'] == 1, color='green', alpha=0.15)

    # draw the plots:
    graph = sns.lineplot(data=df, x=df.index, y=column_name1)
    if column_name2:
        sns.lineplot(data=df, x=df.index, y=column_name2)

    # draw thresholds:
    if thresh1:
        graph.axhline(thresh1, color='red', linewidth=2)
        if thresh2:
            graph.axhline(thresh2, color='red', linewidth=2)
            if thresh3:
                graph.axhline(thresh3, color='blue', linewidth=2)

    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(200))
    plt.xlim(0, df.index[-1])
    plt.grid(True)
    plt.title(title)


# EMA exponential moving average:
def add_exponential_moving_average_EMA_to_df(df, column_name, alpha): #todo check span
    df[column_name + f'_EMA'] = df[column_name].ewm(alpha =alpha).mean()
    return df

def moving_average(df, column_name, window_size):
    # Calculate the moving average using the rolling function
    df[column_name + f'_MA'] = df[column_name].rolling(window=window_size).mean()
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


def get_gnss_heading_360(df_gps):
    '''the heading calculation assumes that the GNSS data is provided in the WGS84 coordinate system or a
    coordinate system where the north direction aligns with the positive y-axis. '''

    # Convert 'latitude' and 'longitude' columns to numeric
    df_gps['latitude'] = pd.to_numeric(df_gps['latitude'], errors='coerce')
    df_gps['longitude'] = pd.to_numeric(df_gps['longitude'], errors='coerce')

    # Calculate the difference in latitude and longitude
    delta_lat = df_gps['latitude'].diff()
    delta_lon = df_gps['longitude'].diff()
    # Calculate the heading using atan2
    heading_rad = np.arctan2(delta_lon, delta_lat)
    # Convert the heading from radians to degrees
    heading_deg = np.degrees(heading_rad)
    # Adjust the heading to be relative to the north
    df_gps['heading_360'] = (heading_deg + 360) % 360
    df_gps['heading_180'] = (heading_deg + 360) % 180
    return df_gps

def gnss_heading_Geodesic(df):
    headings = [None]  # Set the first output row as None
    distances = [None]  # Set the first output row distance as None

    # Iterate over each row of the DataFrame starting from the second row
    for index in range(1, len(df)):
        # Get the latitude and longitude values for the current and previous rows
        lat1 = df.at[index - 1, 'latitude']
        lon1 = df.at[index - 1, 'longitude']
        lat2 = df.at[index, 'latitude']
        lon2 = df.at[index, 'longitude']

        # Calculate the bearing and distance using Geodesic.WGS84.Inverse()
        result = Geodesic.WGS84.Inverse(lat1, lon1, lat2, lon2)
        bearing = result['azi1']
        distance = result['s12']

        # Add the calculated bearing and distance to the respective lists
        headings.append(bearing)
        distances.append(distance)

    # Add the lists of headings and distances as new columns in the DataFrame
    df['bearing'] = headings
    df['distance_meters'] = distances
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
    last_indices = gps_timestamps.searchsorted(merged_timestamps, side='right') - 1
    if np.all(last_indices == -1):
        print("No matching GPS data.")
        return None

    # Filter df_gps using the last indices
    filtered_gps = df_gps.loc[last_indices]

    # Concatenate df_merged and filtered_gps
    merged_df = pd.concat([df_imu.reset_index(drop=True), filtered_gps.reset_index(drop=True)], axis=1)

    return merged_df

def calculate_heading_bounds(expected_heading, threshold):
    # Normalize the headings to be between 0 and 180 degrees
    expected_heading = expected_heading % 180

    # Calculate the lower and upper bounds for the expected heading range
    lower_bound = (expected_heading - threshold) % 180
    upper_bound = (expected_heading + threshold) % 180

    return lower_bound, upper_bound





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

    PATH_ROW = r'/home/matans/Documents/fruitspec/sandbox/Apples_Golan_heights/MED00000/230523/row_1/1'
    PATH_OUTPUT = r'/home/matans/Documents/fruitspec/sandbox/Apples_Golan_heights/MED00000/230523/row_1/1'

    output_dir = os.path.join(PATH_OUTPUT, 'rows_detection')
    df = get_sensors_data(PATH_ROW, output_dir, GT = False, save=True)


    print ('done')








