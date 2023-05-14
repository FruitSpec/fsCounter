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
        df = pd.read_csv(file_path, header=0)
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")
        return None

    return df

def plot_depth_vs_angular_velocity(df, title):
    # Plot:
    plt.figure(figsize=(55, 30))  # Adjust the width and height as desired
    sns.set(font_scale=2)

    # subplot 1
    plt.subplot(5, 1, 1)
    graph1 = sns.lineplot(data=df, x=df.index, y="score")
    #sns.lineplot(data=df, x=df.index, y="score_EMA")
    sns.lineplot(data=df, x=df.index, y="score_EMA")
    graph1.axhline(0.5, color='red', linewidth=2)
    plt.locator_params(axis='y', nbins=11)
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(200))
    plt.xlim(0, df.index[-1])
    plt.grid(True)
    plt.title('Depth score')

    # subplot 2
    plt.subplot(5, 1, 2)
    graph = sns.lineplot(data=df, x=df.index, y="angular_velocity_x")
    sns.lineplot(data=df, x=df.index, y="angular_velocity_x_EMA")
    graph.axhline(10, color='red', linewidth=2)
    graph.axhline(-10, color='red', linewidth=2)
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(200))
    plt.xlim(0, df.index[-1])
    plt.grid(True)
    plt.title('angular_velocity_x')

    # subplot 3
    plt.subplot(5, 1, 3)
    graph = sns.lineplot(data=df, x=df.index, y="heading")
    #sns.lineplot(data=df, x=df.index, y="angular_velocity_x_EMA")
    # graph.axhline(10, color='red', linewidth=2)
    # graph.axhline(-10, color='red', linewidth=2)
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(200))
    plt.xlim(0, df.index[-1])
    plt.grid(True)
    plt.title('heading_gnss')

    # subplot 4
    plt.subplot(5, 1, 4)
    graph = sns.lineplot(data=df, x=df.index, y='abs_delta_heading_north')
    #sns.lineplot(data=df, x=df.index, y="angular_velocity_x_EMA")
    # graph.axhline(10, color='red', linewidth=2)
    # graph.axhline(-10, color='red', linewidth=2)
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(200))
    plt.xlim(0, df.index[-1])
    plt.grid(True)
    plt.title('abs_delta_heading_north')

    # subplot 3
    plt.subplot(5, 1, 5)
    sns.lineplot(data=df, x=df.index, y="GT")
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(200))
    plt.xlim(0, df.index[-1])
    plt.grid(True)
    plt.title('Ground Truth')

    plt.suptitle(title)
    plt.tight_layout()  # Adjust the spacing between subplots
    plt.show()

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
    GT_intervals = [(pd.to_datetime(row.get('start_time'), format='%M:%S'), pd.to_datetime(row.get('end_time'), format='%M:%S')) for row in
                    GT]
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


def gnss_heading(df_gps):
    '''the heading calculation assumes that the GNSS data is provided in the WGS84 coordinate system or a
    coordinate system where the north direction aligns with the positive y-axis. '''

    # Calculate the difference in latitude and longitude
    delta_lat = df_gps['latitude'].diff()
    delta_lon = df_gps['longitude'].diff()
    # Calculate the heading using atan2
    heading_rad = np.arctan2(delta_lon, delta_lat)
    # Convert the heading from radians to degrees
    heading_deg = np.degrees(heading_rad)
    # Adjust the heading to be relative to the north
    heading_deg_adjusted = (heading_deg + 360) % 360
    df_gps['heading'] = heading_deg_adjusted
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


def extract_gnss_data(df_merged, df_gps):
    # TODO - i loose index, if it's needed than should be fixed
    # Extract the timestamp values from both DataFrames
    merged_timestamps = df_merged['timestamp'].values
    gps_timestamps = df_gps['timestamp_gnss'].values

    # Find the indices of the last matching timestamps in df_gps
    last_indices = gps_timestamps.searchsorted(merged_timestamps, side='right') - 1

    # Filter df_gps using the last indices
    filtered_gps = df_gps.loc[last_indices]

    # Concatenate df_merged and filtered_gps
    merged_df = pd.concat([df_merged.reset_index(drop=True), filtered_gps.reset_index(drop=True)], axis=1)

    return merged_df

if __name__ == "__main__":

    # # Load sensor files to df:
    # path_log_imu = r'/home/lihi/FruitSpec/Data/customers/EinVered/SUMERGOL/230423/row_3/imu_1.log'
    # path_depth_csv = r'/home/lihi/FruitSpec/Data/customers/EinVered/SUMERGOL/230423/row_3/rows_detection/depth_ein_vered_SUMERGOL_230423_row_3.csv'
    #
    # df_imu = read_imu_log(path_log_imu)
    # df_depth = pd.read_csv(path_depth_csv, index_col=0).set_index('frame')
    #
    # df_merged2 = pd.concat([df_depth, df_imu], axis=1, join="inner")

    ###########     GNSS     ###################################################
    # Load new csv with sensors data:
    PATH_GPS = r'/home/lihi/FruitSpec/Data/customers/EinVered/SUMERGOL/250423/250423.nav'
    PATH_DEPTH_CSV = r'/home/lihi/FruitSpec/Data/customers/EinVered/SUMERGOL/250423/row_1/rows_detection/depth_ein_vered_SUMERGOL_230423_row_3.csv'
    OUTPUT_DIR = r'/home/lihi/FruitSpec/Data/customers/EinVered/SUMERGOL/250423'
    GT_ROWS_PATH = r'/home/lihi/FruitSpec/Data/customers/EinVered/SUMERGOL/250423/row_1/rows_detection/lines_GT.json'

    df_merged = pd.read_csv(PATH_DEPTH_CSV, index_col=0)
    df_merged.dropna(subset=['score'], inplace=True) #Todo: To remove rows where the 'score' column contains NaN values


    df_gps = read_nav_file(PATH_GPS)

    # Convert the timestamp columns to datetime objects
    df_merged["timestamp"] = pd.to_datetime(df_merged["timestamp"], unit="ns").dt.time
    df_gps["timestamp_gnss"] = pd.to_datetime(df_gps["timestamp"], unit="ns").dt.time
    df_gps.drop('timestamp', axis='columns', inplace=True)

    # calculate heading from gnss data:
    df_gps = gnss_heading(df_gps)

    # ignore directionality (parallel directions should have similar values north to south or south to north)
    condition = df_gps['heading'] > 180
    df_gps['abs_delta_heading_north'] = df_gps['heading']
    df_gps.loc[condition, 'abs_delta_heading_north'] = 360 - df_gps['heading']

    # merge with imu data:
    df_merged = extract_gnss_data(df_merged, df_gps)

    plot_latitude_longitude(df_merged, OUTPUT_DIR, save=False)

    df_merged['timestamp_gnss'] = df_merged['timestamp_gnss'].apply(lambda t: datetime.datetime.combine(datetime.datetime.today(), t))
    # plot_gnss_heading(df_merged, OUTPUT_DIR, column='heading', save=False)
    # plot_gnss_heading(df_merged, OUTPUT_DIR, column='abs_delta_heading_north', save=False)

    # # calc heading using geographiclib:
    # df_gps = gnss_heading_Geodesic(df_gps)
    # plot_gnss_heading(df_gps, OUTPUT_DIR, column='bearing', save=False)



    # calculate EMA:
    df_merged = add_exponential_moving_average_EMA_to_df(df_merged, column_name = 'angular_velocity_x', alpha=0.1) #high alpha is small change
    df_merged = add_exponential_moving_average_EMA_to_df(df_merged, column_name='score', alpha=0.05)

    # df_merged = moving_average(df_merged, column_name ='score', window_size = 30)

    # Tag ground_truth:
    GT_rows = [('0:00', '00:52'), ('1:10', '2:01'), ('02:23', '3:14'), ('3:24', '4:17'), ('4:28', '5:20'),
               ('5:36', '6:33'), ('6:45', '7:37'), ('7:48', '8:41'), ('7:48', '8:41'), ('8:51', '9:43'), ('9:59', '10:53'), ('11:05', '11:58'), ('12:16', '13:08')]

    with open(GT_ROWS_PATH, 'r') as file:
        GT_rows = json.load(file)

    df_merged = GT_to_df(GT_rows,df_merged)

    #plot:
    plot_depth_vs_angular_velocity(df_merged, "EinVered_230423_SUMERGOL_230423_row_2")

    print('done')







