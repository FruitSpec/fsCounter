import pandas as pd
from vision.line_detection.retrieve_sensors_data import plot_sensors
import os
import numpy as np
import cv2
from shapely.geometry import Point, Polygon
from fastkml import kml
import simplekml
from enum import Enum



def update_dataframe_row_results(df, score_new, index, within_row_prediction, pred_changed, row_state, depth_ema, angular_velocity_x_ema, within_row_depth,
                                 within_angular_velocity, within_heading, within_inner_polygon):

    df.at[index, 'score_new'] = score_new
    df.at[index, 'depth_ema'] = depth_ema
    df.at[index, 'ang_vel_ema'] = angular_velocity_x_ema
    df.at[index, 'within_row_depth'] = within_row_depth
    df.at[index, 'within_row_angular_velocity'] = within_angular_velocity
    df.at[index, 'within_row_heading'] = within_heading
    df.at[index, 'within_inner_polygon'] = within_inner_polygon
    df.at[index, 'row_state'] = row_state
    df.at[index, 'pred_changed'] = pred_changed
    df.at[index, 'pred'] = within_row_prediction


    return df

def count_rows(column):
    mask = column == 1
    # Group consecutive "1" periods (cumsum) to assign a unique label to each group
    groups = mask.ne(mask.shift()).cumsum()
    # Filter only the groups where the value is 1 and count them
    rows_count = groups[mask].nunique()
    return rows_count


def generate_new_kml_file(polygon1, polygon2, df, output_dir):
    # Create a new KML document
    kml = simplekml.Kml()

    # Create a new Polygon placemark with style for outline only
    polystyle = simplekml.PolyStyle(fill=0, outline=1)  # No fill, outline only
    linestyle = simplekml.LineStyle(color=simplekml.Color.blue, width=3)  # Blue outline
    style = simplekml.Style()
    style.polystyle = polystyle
    style.linestyle = linestyle

    # First polygon
    placemark1 = kml.newpolygon(name='Polygon1')
    placemark1.outerboundaryis = list(polygon1.exterior.coords)
    placemark1.style = style

    # Second polygon
    placemark2 = kml.newpolygon(name='Polygon2')
    placemark2.outerboundaryis = list(polygon2.exterior.coords)
    placemark2.style = style

    # Sort the DataFrame by timestamp (replace 'timestamp' with the appropriate column)
    df = df.sort_values('timestamp')

    # Create a separate LineString for each pair of points with the same 'pred' value.
    for i in range(len(df) - 1):
        row1 = df.iloc[i]
        row2 = df.iloc[i + 1]
        if row1['pred'] == row2['pred']:
                coords = [(row1['longitude'], row1['latitude']), (row2['longitude'], row2['latitude'])]
                linestring = kml.newlinestring(name=f'LineString{i}')

                linestring.style.linestyle.width = 5

                if row1['pred'] == 0:
                    linestring.style.linestyle.color = simplekml.Color.red  # Red for pred = 0
                else:
                    linestring.style.linestyle.color = simplekml.Color.yellow  # Yellow for pred = 1
                linestring.coords = coords

    # Add labels to the start and end points of the line
    start_label = kml.newpoint(name='Start', coords=[(df.iloc[0]['longitude'], df.iloc[0]['latitude'])])
    end_label = kml.newpoint(name='End', coords=[(df.iloc[-1]['longitude'], df.iloc[-1]['latitude'])])

    # Save the KML document to a new file
    output_path = os.path.join(output_dir, 'new_file.kml')
    kml.save(output_path)
    print(f'KML file saved to {output_path}')



class RowState(Enum):
    NOT_IN_ROW = 0
    STARTING_ROW = 1
    MIDDLE_OF_ROW = 2
    ENDING_ROW = 3

class RowDetector:

    def __init__(self, path_kml, placemark_name, expected_heading):

        # Constants:
        self.path_kml = path_kml
        self.placemark_name = placemark_name
        self.MARGINS_THRESHOLD = 3
        self.polygon = self.parse_kml_file()
        self.EXPECTED_HEADING = expected_heading
        self.HEADING_THRESHOLD = 30
        self.DEPTH_WINDOW_Y_HIGH = 0.35
        self.DEPTH_WINDOW_Y_LOW = 0.75
        self.DEPTH_THRESHOLD = 0.5
        self.DEPTH_EMA_ALPHA = 0.02
        self.ANGULAR_VELOCITY_THRESHOLD = 10
        self.ANGULAR_VELOCITY_EMA_ALPHA = 0.02
        self.CONSISTENCY_THRESHOLD = 3
        self.lower_bound, self.upper_bound = self.get_heading_bounds_180(self.EXPECTED_HEADING, self.HEADING_THRESHOLD)

        # Init:
        self.inner_polygon = self.get_inner_polygon(self.polygon, self.MARGINS_THRESHOLD)
        self.depth_ema = 0.5
        self.angular_velocity_x_ema = 0
        self.consistency_counter = 0
        self.previous_longitude = None
        self.previous_latitude = None
        self.depth_score = None
        self.within_row_angular_velocity = None
        self.within_row_depth = None
        self.within_row_heading = None
        self.within_inner_polygon = None
        self.row_state = RowState.NOT_IN_ROW
        self.row_pred = None
        self.pred_changed = False


    def global_decision(self):
        # Reset state_changed to False at the start of each call
        self.pred_changed = False

        # State: Not_in_Row
        if self.row_state == RowState.NOT_IN_ROW:
            if self.within_row_depth and self.within_row_heading:               # if depth + heading => enter row
                self.consistency_counter += 1
                if self.consistency_counter >= self.CONSISTENCY_THRESHOLD:
                    self.row_state = RowState.STARTING_ROW
                    self.consistency_counter = 0
                    self.pred_changed = True
            else:
                self.consistency_counter = 0

        # State: Starting a Row
        elif self.row_state == RowState.STARTING_ROW:
            if self.within_inner_polygon:
                self.consistency_counter += 1
                if self.consistency_counter >= self.CONSISTENCY_THRESHOLD:
                    self.row_state = RowState.MIDDLE_OF_ROW
                    self.consistency_counter = 0
            else:
                self.consistency_counter = 0

        # State: Middle of a Row
        elif self.row_state == RowState.MIDDLE_OF_ROW:
            if self.within_inner_polygon:             # todo: add delay?
                self.consistency_counter = 0
            else:
                self.consistency_counter += 1
                if self.consistency_counter >= self.CONSISTENCY_THRESHOLD:
                    self.row_state = RowState.ENDING_ROW
                    self.consistency_counter = 0


        # State: Ending a Row
        elif self.row_state == RowState.ENDING_ROW:
            if self.within_inner_polygon:
                self.row_state = RowState.MIDDLE_OF_ROW
            else:
                if not self.within_row_angular_velocity:
                    self.consistency_counter += 1
                    if self.consistency_counter >= self.CONSISTENCY_THRESHOLD:
                        self.row_state = RowState.NOT_IN_ROW
                        self.consistency_counter = 0
                        self.pred_changed = True
                else:
                    self.consistency_counter = 0



    def sensors_decision(self, angular_velocity_x, longitude, latitude):
        # depth sensor:
        self.depth_ema = self.exponential_moving_average(self.depth_score, self.depth_ema, alpha=self.DEPTH_EMA_ALPHA)
        self.within_row_depth = self.depth_ema <= self.DEPTH_THRESHOLD

        # jairo's sensor:
        self.angular_velocity_x_ema = self.exponential_moving_average(angular_velocity_x, self.angular_velocity_x_ema, alpha=self.ANGULAR_VELOCITY_EMA_ALPHA)
        self.within_row_angular_velocity = abs(self.angular_velocity_x_ema) < self.ANGULAR_VELOCITY_THRESHOLD

        # gnss sensor:
        heading_360,heading_180 =  self.get_heading(longitude, latitude)
        if heading_360 and heading_180 is not None:   # The first time the heading is None
            self.within_row_heading = self.heading_within_range(heading_180, self.lower_bound, self.upper_bound)
        self.within_inner_polygon = self.is_within_inner_polygon((longitude, latitude))

    def detect_row(self,  angular_velocity_x, longitude , latitude, rgb_img = None, depth_img = None, depth_score = None):

        if depth_img is not None:   # Todo: remove 'if'. Currently for debugging with depth score from csv
            self.percent_far_pixels(depth_img, rgb_img = rgb_img)
        self.sensors_decision(angular_velocity_x, longitude , latitude)
        self.global_decision()

        self.row_pred = int(self.row_state != RowState.NOT_IN_ROW)
        return self.row_pred, self.pred_changed


    def get_heading(self, longitude_curr, latitude_curr):
        '''the heading calculation assumes that the GNSS data is provided in the WGS84 coordinate system or a
        coordinate system where the north direction aligns with the positive y-axis. '''

        if self.previous_latitude and self.previous_longitude:  # The first time the heading is None
            # Calculate the difference in latitude and longitude
            delta_lat = latitude_curr - self.previous_latitude
            delta_lon = longitude_curr - self.previous_longitude
            # Calculate the heading using atan2
            heading_rad = np.arctan2(delta_lon, delta_lat)
            # Convert the heading from radians to degrees
            heading_deg = np.degrees(heading_rad)
            # Adjust the heading to be relative to the north
            heading_360 = (heading_deg + 360) % 360
            heading_180 = (heading_deg + 360) % 180

            self.previous_latitude = latitude_curr
            self.previous_longitude = longitude_curr
            return heading_360, heading_180
        else:
            self.previous_latitude = latitude_curr
            self.previous_longitude = longitude_curr
            return None, None

    def parse_kml_file(self):
        # Check if file exists
        if not os.path.isfile(self.path_kml):
            raise Exception(f"File {self.file_path} not found.")
        with open(self.path_kml, 'rt', encoding="utf-8") as file:
            doc = file.read()

        k = kml.KML()
        k.from_string(doc)

        # Retrieve Placemarks from KML
        features = list(k.features())
        f2 = list(features[0].features())

        for placemark in f2:
            if placemark.name == self.placemark_name:
                if placemark.geometry.geom_type == 'Polygon':
                    polygon_geom = placemark.geometry
                    return Polygon(polygon_geom.exterior.coords)

        raise Exception(f"Placemark with name {self.placemark_name} containing a Polygon not found.")

    def get_inner_polygon(self, polygon, margins_meters):
        # Convert self.MARGINS_THRESHOLD from meters to degrees (assuming a flat Earth approximation)
        degrees_per_meter = 1 / 111000
        buffer_distance_deg = margins_meters * degrees_per_meter

        # Create inner polygon by buffering original polygon with the converted distance
        inner_polygon = polygon.buffer(-buffer_distance_deg)

        # If the buffering operation results in multiple polygons, we take the largest one
        if inner_polygon.type == 'MultiPolygon':
            inner_polygon = max(inner_polygon, key=lambda x: x.area)
        return inner_polygon


    def is_within_inner_polygon(self, gnss_position):
        point = Point(gnss_position)
        self.within_inner_polygon = self.inner_polygon.contains(point)
        return self.within_inner_polygon

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

    def percent_far_pixels(self, depth_img, rgb_img):
        # todo: un-comment
        # todo: check if RGB or BGR
        # # shadow sky noise:
        # blue = rgb_img[:, :, 0].copy()
        # depth[blue > 240] = 0

        y_high = int(depth_img.shape[0] * self.DEPTH_WINDOW_Y_HIGH)
        y_low = int(depth_img.shape[0] * self.DEPTH_WINDOW_Y_LOW)

        width = depth_img.shape[1]
        search_area = depth_img[y_high: y_low, :].copy()
        score = np.sum(search_area < 20) / ((y_low - y_high) * width) # % pixels blow threshold
        self.depth_score = round(score, 2)

if __name__ == '__main__':

    CSV_PATH = r'/home/lihi/FruitSpec/Data/customers/EinVered/SUMERGOL/250423/row_3/rows_detection/sensors_EinVered_SUMERGOL_250423_row_3.csv'
    DEPTH_VIDEO_PATH = r'/home/lihi/FruitSpec/Data/customers/EinVered/SUMERGOL/250423/row_3/rows_detection/depth_draw_EinVered_SUMERGOL_250423_row_3.mp4'
    RGB_VIDEO_PATH = r'/home/lihi/FruitSpec/Data/customers/EinVered/SUMERGOL/250423/row_3/RGB_1.mkv'
    PATH_KML = r'/home/lihi/FruitSpec/Data/customers/EinVered/SUMERGOL/Blocks.kml'
    EXPECTED_HEADING = 100

    PATH_ROW = os.path.dirname(os.path.dirname(CSV_PATH))
    output_dir = os.path.join(PATH_ROW, 'rows_detection')
    output_name = "_".join(PATH_ROW.split('/')[-4:])

    df = pd.read_csv(CSV_PATH)
    print(f'Loaded {CSV_PATH}')

    # init rows detector:
    row_detector = RowDetector(expected_heading = EXPECTED_HEADING, path_kml = PATH_KML, placemark_name = 'SUMERGOL')

    # load video:
    cap_rgb = cv2.VideoCapture(RGB_VIDEO_PATH)
    cap_depth = cv2.VideoCapture(DEPTH_VIDEO_PATH)
    print(f'Loaded {DEPTH_VIDEO_PATH}')
    print (f'Video_len:{int(cap_depth.get(cv2.CAP_PROP_FRAME_COUNT))}, df_len:{len(df)}')

    # load frames:
    for index, row in df.iterrows():

        # Get depth and RGB frames:
        ret1, depth_img = cap_depth.read()
        if not ret1:
            break
        ret2, rgb_img = cap_rgb.read()
        if not ret2:
            break

        depth_img = depth_img[:, :, 0].copy()

        is_row, state_changed = row_detector.detect_row(depth_img=depth_img, rgb_img=rgb_img, angular_velocity_x = row.angular_velocity_x, longitude = row.longitude, latitude = row.latitude)
        print(f'{index}: {is_row}_{state_changed}')
        # if score from csv:
        #is_row = row_detector.detect_row(depth_img=None, rgb_img=None, depth_score = row['score'], angular_velocity_x=row['angular_velocity_x'])

        # within_row_prediction = 1 if is_row ==  "In_Row" else 0

        # Update dataframe with results:
        df = update_dataframe_row_results(df = df, score_new = row_detector.depth_score, index = index, within_row_prediction = row_detector.row_pred, pred_changed = row_detector.pred_changed, row_state = row_detector.row_state.value ,depth_ema = row_detector.depth_ema, angular_velocity_x_ema = row_detector.angular_velocity_x_ema, within_row_depth = row_detector.within_row_depth, within_angular_velocity = row_detector.within_row_angular_velocity, within_heading = row_detector.within_row_heading,  within_inner_polygon = row_detector.within_inner_polygon)


    rows_in_GT = count_rows(column= df['GT'])
    rows_in_Pred = count_rows(column=df['pred'])

    generate_new_kml_file(row_detector.polygon, row_detector.inner_polygon,df, output_dir = output_dir)

    # plot sensors data:
    config = f'EX1_GT:{rows_in_GT}_Pred:{rows_in_Pred}_Enter_depth_and_heading__Exit_ang_vel_DEPTH_THRESH {row_detector.DEPTH_THRESHOLD}, DEPTH_EMA {row_detector.DEPTH_EMA_ALPHA}, ANG_VEL_THRESH {row_detector.ANGULAR_VELOCITY_THRESHOLD}, ANG_VEL_EMA {row_detector.ANGULAR_VELOCITY_EMA_ALPHA}, EXPECTED_HEADING {row_detector.EXPECTED_HEADING}, HEADING_THRESH {row_detector.HEADING_THRESHOLD}_, self.MARGINS_THRESHOLD {row_detector.MARGINS_THRESHOLD}'

    plot_sensors(df, config + output_name,
                 depth_threshold = row_detector.DEPTH_THRESHOLD,
                 angular_velocity_threshold = row_detector.ANGULAR_VELOCITY_THRESHOLD,
                 expected_heading = row_detector.EXPECTED_HEADING,
                 lower_heading_bound = row_detector.lower_bound,
                 upper_heading_bound= row_detector.upper_bound,
                 margins_threshold = row_detector.MARGINS_THRESHOLD,
                 save_dir = output_dir)

    print('Done!')
