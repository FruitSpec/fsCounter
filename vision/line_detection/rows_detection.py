import pandas as pd
from vision.line_detection.retrieve_sensors_data import plot_sensors
import os
import numpy as np
import cv2

def update_dataframe_row_results(df, index, within_row_prediction, depth_ema, angular_velocity_x_ema, within_row_depth,
                                 within_row_angular_velocity, within_row_heading):
    print(f'Frame {index}: {within_row_prediction}')
    df.at[index, 'pred'] = within_row_prediction
    df.at[index, 'depth_ema'] = depth_ema
    df.at[index, 'ang_vel_ema'] = angular_velocity_x_ema
    df.at[index, 'within_row_depth'] = within_row_depth
    df.at[index, 'within_row_angular_velocity'] = within_row_angular_velocity
    df.at[index, 'within_row_heading'] = within_row_heading
    return df

def count_rows(column):
    mask = column == 1
    # Group consecutive "1" periods (cumsum) to assign a unique label to each group
    groups = mask.ne(mask.shift()).cumsum()
    # Filter only the groups where the value is 1 and count them
    rows_count = groups[mask].nunique()
    return rows_count


class RowDetector:

    def __init__(self, expected_heading):

        # Constants:
        self.EXPECTED_HEADING = expected_heading
        self.HEADING_THRESHOLD = 30
        self.DEPTH_THRESHOLD = 0.5
        self.DEPTH_EMA_ALPHA = 0.02
        self.ANGULAR_VELOCITY_THRESHOLD = 10
        self.ANGULAR_VELOCITY_EMA_ALPHA = 0.02
        self.CONSISTENCY_THRESHOLD = 3
        self.lower_bound, self.upper_bound = self.get_heading_bounds_180(self.EXPECTED_HEADING, self.HEADING_THRESHOLD)

        # Init:
        self.depth_ema = 0.5
        self.angular_velocity_x_ema = 0
        self.consistency_counter = 0
        self.state = "Not_in_Row"

        self.within_row_angular_velocity = None
        self.within_row_depth = None
        self.within_row_heading = None



    def global_decision(self):
        # Enter a row:
        if self.state == "Not_in_Row":
            if self.within_row_depth and self.within_row_heading:
                self.consistency_counter += 1
                if self.consistency_counter >= self.CONSISTENCY_THRESHOLD:
                    self.state = "In_Row"
                    self.consistency_counter = 0
            else:
                self.consistency_counter = 0
        # Exit a row:
        elif self.state == "In_Row":
            if not self.within_row_angular_velocity:
                self.consistency_counter += 1
                if self.consistency_counter >= self.CONSISTENCY_THRESHOLD:
                    self.state = "Not_in_Row"
                    self.consistency_counter = 0
            else:
                self.consistency_counter = 0


    def sensors_decision(self, depth_score, angular_velocity_x, heading_180):

        self.depth_ema = self.exponential_moving_average(depth_score, self.depth_ema, alpha=self.DEPTH_EMA_ALPHA)
        self.within_row_depth = self.depth_ema <= self.DEPTH_THRESHOLD

        self.angular_velocity_x_ema = self.exponential_moving_average(angular_velocity_x, self.angular_velocity_x_ema, alpha=self.ANGULAR_VELOCITY_EMA_ALPHA)
        self.within_row_angular_velocity = abs(self.angular_velocity_x_ema) < self.ANGULAR_VELOCITY_THRESHOLD

        self.within_row_heading = self.heading_within_range(heading_180, self.lower_bound, self.upper_bound)

    def detect_row(self, depth_score, angular_velocity_x, heading_180):
        self.sensors_decision(depth_score, angular_velocity_x, heading_180)
        self.global_decision()
        return self.state

    @staticmethod
    def heading_within_range(current_heading, lower_bound, upper_bound):

        # Check if the current heading falls within the expected heading range
        if lower_bound <= upper_bound:
            heading_within_range = lower_bound <= current_heading <= upper_bound
        else:
            # Handle the case where the expected heading range wraps around 360 degrees
            heading_within_range = current_heading >= lower_bound or current_heading <= upper_bound

        return heading_within_range
    @staticmethod
    def get_heading_bounds_180(expected_heading, threshold):
        # Normalize the headings to be between 0 and 180 degrees
        expected_heading = expected_heading % 180

        # Calculate the lower and upper bounds for the expected heading range
        lower_bound = (expected_heading - threshold) % 180
        upper_bound = (expected_heading + threshold) % 180

        return lower_bound, upper_bound

    @staticmethod
    def exponential_moving_average(x, last_ema, alpha=0.5):
        return alpha * x + (1 - alpha) * last_ema


if __name__ == '__main__':

    CSV_PATH = r'/home/lihi/FruitSpec/Data/customers/EinVered/SUMERGOL/250423/row_3/rows_detection/sensors_EinVered_SUMERGOL_250423_row_3.csv'
    DEPTH_VIDEO_PATH = r'/home/lihi/FruitSpec/Data/customers/EinVered/SUMERGOL/250423/row_3/rows_detection/depth_draw_EinVered_SUMERGOL_250423_row_3.mp4'
    EXPECTED_HEADING = 100

    PATH_ROW = os.path.dirname(os.path.dirname(CSV_PATH))
    output_dir = os.path.join(PATH_ROW, 'rows_detection')
    output_name = "_".join(PATH_ROW.split('/')[-4:])

    df = pd.read_csv(CSV_PATH)
    print(f'Loaded {CSV_PATH}')

    # init rows detector:
    row_detector = RowDetector(expected_heading = EXPECTED_HEADING)

    # load video:
    cap = cv2.VideoCapture(DEPTH_VIDEO_PATH)
    print(f'Loaded {DEPTH_VIDEO_PATH}')
    print (f'Video_len:{int(cap.get(cv2.CAP_PROP_FRAME_COUNT))}, df_len:{len(df)}')

    # load frames:
    for index, row in df.iterrows():
        ########################################################
        # Get frame:
        ret, img = cap.read()
        if not ret:
            break

        ######################################################
        is_row = row_detector.detect_row(row['score'], row['angular_velocity_x'], row['heading_180'])
        within_row_prediction = 1 if is_row == "In_Row" else 0
        print (f'Frame {index}: {within_row_prediction}_{is_row}')


        # Update dataframe with results:
        df = update_dataframe_row_results(df, index, within_row_prediction, row_detector.depth_ema, row_detector.angular_velocity_x_ema, row_detector.within_row_depth, row_detector.within_row_angular_velocity, row_detector.within_row_heading)


    rows_in_GT = count_rows(column= df['GT'])
    rows_in_Pred = count_rows(column=df['pred'])

    # plot sensors data:
    config = f'EX1_GT:{rows_in_GT}_Pred:{rows_in_Pred}_Enter_depth_and_heading__Exit_ang_vel_DEPTH_THRESH {row_detector.DEPTH_THRESHOLD}, DEPTH_EMA {row_detector.DEPTH_EMA_ALPHA}, ANG_VEL_THRESH {row_detector.ANGULAR_VELOCITY_THRESHOLD}, ANG_VEL_EMA {row_detector.ANGULAR_VELOCITY_EMA_ALPHA}, EXPECTED_HEADING {row_detector.EXPECTED_HEADING}, HEADING_THRESH {row_detector.HEADING_THRESHOLD}_'

    plot_sensors(df, config + output_name,
                 depth_threshold = row_detector.DEPTH_THRESHOLD,
                 angular_velocity_threshold = row_detector.ANGULAR_VELOCITY_THRESHOLD,
                 expected_heading = row_detector.EXPECTED_HEADING,
                 lower_heading_bound = row_detector.lower_bound,
                 upper_heading_bound= row_detector.upper_bound,
                 save_dir = output_dir)

    print('Done!')
