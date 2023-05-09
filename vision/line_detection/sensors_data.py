import logging

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np
import os

def load_log_file(file_path):
    try:
        with open(file_path, 'r') as file:
            log_data = file.readlines()
        return log_data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except Exception as e:
        print(f"An error occurred while loading the log file: {e}")
        return None

def log_to_df(log_data):
    table_data = []
    current_row = []

    for line in log_data:
        line = line.strip()  # Remove leading/trailing whitespace
        if line:
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
        return table
    else:
        return None


def plot_depth_vs_angular_velocity(df, title):
    # Plot:
    plt.figure(figsize=(50, 25))  # Adjust the width and height as desired
    sns.set(font_scale=2)
    # subplot 1
    plt.subplot(3, 1, 1)
    graph1 = sns.lineplot(data=df, x=df.index, y="score")
    #sns.lineplot(data=df, x=df.index, y="score_EMA")
    sns.lineplot(data=df, x=df.index, y="score_MA")
    graph1.axhline(0.5, color='red', linewidth=2)
    plt.locator_params(axis='y', nbins=11)
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(200))
    plt.xlim(0, df.index[-1])
    plt.grid(True)
    plt.title('Depth score')

    # subplot 2
    plt.subplot(3, 1, 2)
    graph = sns.lineplot(data=df, x=df.index, y="angular_velocity_x")
    sns.lineplot(data=df, x=df.index, y="angular_velocity_x_EMA")
    graph.axhline(10, color='red', linewidth=2)
    graph.axhline(-10, color='red', linewidth=2)
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(200))
    plt.xlim(0, df.index[-1])
    plt.grid(True)
    plt.title('angular_velocity_x')

    # subplot 2
    plt.subplot(3, 1, 3)
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
    GT_intervals = [(pd.to_datetime(start, format='%M:%S'), pd.to_datetime(end, format='%M:%S')) for start, end in
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


if __name__ == "__main__":

    # Load log file to df:
    log_file_path = f'/home/lihi/FruitSpec/Data/customers/EinVered/230423/SUMERGOL/230423/row_3/imu_1.log'

    log_contents = load_log_file(log_file_path)
    df_imu = log_to_df(log_contents)
    df_imu.columns = ['date', 'timestamp', 'angular_velocity_x', 'angular_velocity_y', 'angular_velocity_z', 'linear_acceleration_x', 'linear_acceleration_y', 'linear_acceleration_z']
    df_imu = extract_time_from_timestamp(df_imu)

    # Load depth cvs file and merge with sensors:
    path_to_depth = r'/home/lihi/FruitSpec/debbug/depth_ein_vered_SUMERGOL_230423_row_3_.csv'
    df_depth = pd.read_csv(path_to_depth, index_col=0).set_index('frame')
    df_merged = pd.concat([df_depth, df_imu], axis=1, join="inner")

    # calculate EMA:
    df_merged = add_exponential_moving_average_EMA_to_df(df_merged, column_name = 'angular_velocity_x', alpha=0.05) #high alpha is small change
    df_merged = add_exponential_moving_average_EMA_to_df(df_merged, column_name='score', alpha=0.01)

    df_merged = moving_average(df_merged, column_name ='score', window_size = 30)

    # Tag ground_truth:
    GT_rows = [('0:21', '1:20'), ('1:35', '2:25'), ('2:36', '3:28'), ('3:43', '4:37'), ('4:52', '5:42'), ('5:53','6:46')]
    df_merged = GT_to_df(GT_rows,df_merged)

    #plot:
    plot_depth_vs_angular_velocity(df_merged, "EinVered_230423_SUMERGOL_230423_row_3")

    # todo - i got drift in ground truth results since of the bug that i had to remove every 30 frames




