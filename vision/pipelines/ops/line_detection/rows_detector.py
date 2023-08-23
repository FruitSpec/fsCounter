import pandas as pd
import os
import numpy as np
import cv2
from shapely.geometry import Point, Polygon
from fastkml import kml
import simplekml
from enum import Enum
from math import asin, atan2, cos, degrees, radians, sin, sqrt
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from skimage._shared.utils import deprecate_func


class RowState(Enum):
    NOT_IN_ROW = 0
    STARTING_ROW = 1
    MIDDLE_OF_ROW = 2
    ENDING_ROW = 3


class RowDetector:

    def __init__(self, path_kml, placemark_name, expected_heading=None):

        # Constants:
        self.path_kml = path_kml
        self.placemark_name = placemark_name
        self.MARGINS_THRESHOLD = 6
        self.polygon = self.parse_kml_file()
        self.EXPECTED_HEADING = expected_heading
        self.rows_entry_polygons = None
        self.HEADING_THRESHOLD = 30
        self.DEPTH_WINDOW_Y_HIGH = 0.35
        self.DEPTH_WINDOW_Y_LOW = 0.65
        self.DEPTH_THRESHOLD = 0.5
        self.DEPTH_EMA_ALPHA = 0.02
        self.ANGULAR_VELOCITY_THRESHOLD = 8  #10
        self.ANGULAR_VELOCITY_EMA_ALPHA = 0.02
        self.CONSISTENCY_THRESHOLD = 3
        self.lower_boundF, self.upper_boundF = None, None
        if self.EXPECTED_HEADING is not None:
            self.lower_boundF, self.upper_boundF, self.lower_boundR, self.upper_boundR = self.get_heading_bounds(
                self.EXPECTED_HEADING, self.HEADING_THRESHOLD)

            self.rows_entry_polygons = self.get_rows_entry_polygons()

        # Init:
        self.inner_polygon = RowDetector.get_inner_polygon(self.polygon, self.MARGINS_THRESHOLD)
        self.depth_ema = 0.5
        self.angular_velocity_x_ema = 0
        self.heading_360F, self.heading_360R = None, None
        self.consistency_counter = 0
        self.previous_longitude = None
        self.previous_latitude = None
        self.depth_score = None
        self.within_row_angular_velocity = None
        self.within_row_depth = None
        self.within_row_heading = None
        self.within_inner_polygon = None
        self.within_rows_entry_polygons = None
        self.row_state = RowState.NOT_IN_ROW
        self.row_pred = None
        self.pred_changed = False
        self.point_inner_polygon_in = None
        self.point_inner_polygon_out = None
        self.point_start_row = None
        self.point_end_of_row = None
        self.row_number = int(0)
        self.row_length = None
        self.df = pd.DataFrame()
        self.index = 0

    def get_rows_entry_polygons(self):
        self.rows_entry_polygons = []

        # calculate heading for polygon edges:
        coords = list(self.polygon.exterior.coords)
        for i in range(len(coords) - 1):
            lon1, lat1 = coords[i]
            lon2, lat2 = coords[i + 1]

            heading_360F, heading_360R, _, _ = self.get_heading(lon2, lat2, lon1, lat1)

            lower_boundF, upper_boundF, lower_boundR, upper_boundR = self.get_heading_bounds(
                self.EXPECTED_HEADING, 25)

            within_heading = RowDetector.heading_within_range_FR(heading_360F, lower_boundF, upper_boundF, lower_boundR, upper_boundR)

            # if the heading is not within range, create a new polygon
            if not within_heading:
                p1_lat1, p1_lon1 = RowDetector.get_point_at_distance(lat1, lon1, self.MARGINS_THRESHOLD,
                                                                     self.EXPECTED_HEADING)
                p1_lat2, p1_lon2 = RowDetector.get_point_at_distance(lat1, lon1, self.MARGINS_THRESHOLD,
                                                                     self.EXPECTED_HEADING + 180)
                p2_lat1, p2_lon1 = RowDetector.get_point_at_distance(lat2, lon2, self.MARGINS_THRESHOLD,
                                                                     self.EXPECTED_HEADING)
                p2_lat2, p2_lon2 = RowDetector.get_point_at_distance(lat2, lon2, self.MARGINS_THRESHOLD,
                                                                     self.EXPECTED_HEADING + 180)

                # generate new polygon from points:
                new_polygon = Polygon([(p1_lon1, p1_lat1), (p1_lon2, p1_lat2), (p2_lon2, p2_lat2), (p2_lon1, p2_lat1)])
                self.rows_entry_polygons.append(new_polygon)

        return self.rows_entry_polygons

    def _get_observed_row_heading_and_update_expected_heading(self):
        '''
        # 1. calculate heading when exiting row
        # 2. if heading is unknown: update expected heading
        '''
        # Calculate heading when exiting row:
        self.row_heading_360F, row_heading_360R,  _, _ = self.get_heading(
            self.point_inner_polygon_out[0], self.point_inner_polygon_out[1], self.point_inner_polygon_in[0],
            self.point_inner_polygon_in[1])
        # if heading is unknown: update expected heading from the first row
        if self.EXPECTED_HEADING is None:
            self.EXPECTED_HEADING = self.row_heading_360F
            self.lower_boundF, self.upper_boundF, self.lower_boundR, self.upper_boundR = self.get_heading_bounds(self.EXPECTED_HEADING, self.HEADING_THRESHOLD)

            self.rows_entry_polygons = self.get_rows_entry_polygons()

    def global_decision(self, longitude , latitude):

        # Reset:
        self.pred_changed = False
        self.row_heading_360F = None
        self.row_length = None

        # State: Not_in_Row
        if self.row_state == RowState.NOT_IN_ROW:
            # if heading is unknown:  + depth + angular velocity + inner polygon => enter row
            # if heading is known:  + depth + heading => enter row
            conditions_to_enter_row_unknown_heading = (self.EXPECTED_HEADING is None) and self.within_row_depth and self.within_row_angular_velocity and self.within_inner_polygon
            conditions_to_enter_row_known_heading = self.within_row_depth and self.within_row_heading
            if conditions_to_enter_row_unknown_heading or conditions_to_enter_row_known_heading:
                self.row_state = RowState.STARTING_ROW
                self.row_number += 1
                self.pred_changed = True
                self.point_start_row = (longitude, latitude)


        # State: Starting a Row
        elif self.row_state == RowState.STARTING_ROW:
            # If heading is unknown: + inner polygon => middle of row:
            # If heading is known: + not in rows entry polygons => middle of row:
            conditions_to_middle_row_unknown_heading = (self.within_rows_entry_polygons is None) and self.within_inner_polygon
            conditions_to_middle_row_known_heading = (self.within_rows_entry_polygons is not None) and (not self.within_rows_entry_polygons)
            if conditions_to_middle_row_unknown_heading or conditions_to_middle_row_known_heading:
                self.row_state = RowState.MIDDLE_OF_ROW
                self.point_inner_polygon_in = (longitude, latitude)


        # State: Middle of a Row
        elif self.row_state == RowState.MIDDLE_OF_ROW:
            # If heading is unknown: + not inner polygon => end row
            # If heading is known: + within_rows_entry_polygons => end row
            conditions_to_end_row_unknown_heading = (self.within_rows_entry_polygons is None) and (not self.within_inner_polygon)
            conditions_to_end_row_known_heading = (self.within_rows_entry_polygons is not None) and (self.within_rows_entry_polygons)
            if conditions_to_end_row_unknown_heading or conditions_to_end_row_known_heading:
                self.row_state = RowState.ENDING_ROW
                self.point_inner_polygon_out = (longitude, latitude)


        # State: Ending a Row
        elif self.row_state == RowState.ENDING_ROW:
            # If heading is unknown: + in_inner_polygon => MIDDLE_OF_ROW
            # If heading is known: + not in_rows_entry_polygons => MIDDLE_OF_ROW
            conditions_back_to_MIDDLE_OF_ROW_unknown_heading = (self.within_rows_entry_polygons is None) and (self.within_inner_polygon)
            conditions_back_to_MIDDLE_OF_ROW_known_heading = (self.within_rows_entry_polygons is not None) and (not self.within_rows_entry_polygons)
            if conditions_back_to_MIDDLE_OF_ROW_unknown_heading or conditions_back_to_MIDDLE_OF_ROW_known_heading:
                self.row_state = RowState.MIDDLE_OF_ROW


            # If angular velocity and ENDING_ROW => NOT_IN_ROW
            conditions_to_NOT_IN_ROW = (self.row_state == RowState.ENDING_ROW) and (not self.within_row_angular_velocity)
            if conditions_to_NOT_IN_ROW:
                self.row_state = RowState.NOT_IN_ROW
                self.pred_changed = True
                self.point_end_of_row = (longitude, latitude)
                self.row_length = RowDetector.get_distance(lon1=longitude, lat1=latitude, lon2=self.point_start_row[0], lat2=self.point_start_row[1])

                # update current row heading and expected heading:
                self._get_observed_row_heading_and_update_expected_heading()

    def sensors_decision(self, angular_velocity_x, longitude, latitude):

        # depth sensor:
        self.depth_ema = RowDetector.exponential_moving_average(self.depth_score, self.depth_ema, alpha=self.DEPTH_EMA_ALPHA)
        self.within_row_depth = self.depth_ema <= self.DEPTH_THRESHOLD

        # gyro sensor:
        self.angular_velocity_x_ema = RowDetector.exponential_moving_average(angular_velocity_x, self.angular_velocity_x_ema, alpha=self.ANGULAR_VELOCITY_EMA_ALPHA)
        self.within_row_angular_velocity = abs(self.angular_velocity_x_ema) < self.ANGULAR_VELOCITY_THRESHOLD

        # gnss sensor:
        self.heading_360F, self.heading_360R, self.previous_longitude, self.previous_latitude  =  self.get_heading(longitude, latitude, self.previous_longitude, self.previous_latitude)

        #  Check if the heading is within the range
        if (self.heading_360F is not None) and (self.heading_360R is not None) and (self.lower_boundF is not None) and (self.upper_boundF is not None):
            self.within_row_heading = RowDetector.heading_within_range_FR(self.heading_360R, self.lower_boundF, self.upper_boundF, self.lower_boundR,
                                                          self.upper_boundR) # will be None if heading is None

        # Check if the current location is within the row entry polygons and inner polygon
        self.within_rows_entry_polygons = self.point_within_rows_entry_polygons((longitude, latitude))
        self.within_inner_polygon = self.is_within_inner_polygon((longitude, latitude))

    def detect_row(self,  angular_velocity_x, longitude, latitude, imu_timestamp, gps_timestamp,
                   ground_truth=None, rgb_img=None, depth_img=None, depth_score=None):

        if depth_score:
            self.depth_score = depth_score
        elif depth_img:
            self.depth_score = RowDetector.percent_far_pixels(
                depth_img=depth_img, rgb_img=rgb_img, show_video=False,
                depth_window_y_high=self.DEPTH_WINDOW_Y_HIGH, depth_window_y_low=self.DEPTH_WINDOW_Y_LOW
            )
        if bool(depth_score) == bool(depth_img):
            raise ValueError("detect_row function must get either depth_score or depth_img")

        self.sensors_decision(angular_velocity_x, longitude, latitude)
        self.global_decision(longitude, latitude)

        self.row_pred = int(self.row_state != RowState.NOT_IN_ROW)
        self.update_dataframe_results(imu_timestamp, angular_velocity_x, gps_timestamp, longitude, latitude,
                                      ground_truth)
        self.index += 1
        return self.row_pred, self.pred_changed, self.df

    def get_heading(self, longitude_curr, latitude_curr, longitude_previous, latitude_previous):
        """
        the heading calculation assumes that the GNSS data is provided in the WGS84 coordinate system or a
        coordinate system where the north direction aligns with the positive y-axis.
        """

        if latitude_previous and longitude_previous:  # The first time the heading is None
            if (longitude_curr == longitude_previous) and (latitude_curr == latitude_previous): # If the location is the same as the previous one, keep the same heading
                return self.heading_360F, self.heading_360R, longitude_curr, latitude_curr

            else: # If the location is different from the previous one, calculate the heading:
                # Calculate the difference in latitude and longitude
                delta_lat = latitude_curr - latitude_previous
                delta_lon = longitude_curr - longitude_previous
                # Calculate the heading using atan2
                heading_rad = np.arctan2(delta_lon, delta_lat)
                # Convert the heading from radians to degrees
                heading_deg = np.degrees(heading_rad)
                # Adjust the heading to be relative to the north
                heading_360F = (heading_deg + 360) % 360
                heading_360R = (heading_deg + 180) % 360

                return heading_360F, heading_360R, longitude_curr, latitude_curr

        else:
            return None, None, longitude_curr, latitude_curr

    def parse_kml_file(self):
        # Check if file exists
        if not os.path.isfile(self.path_kml):
            raise Exception(f"File {self.path_kml} not found.")
        with open(self.path_kml, 'rt', encoding="utf-8") as f:
            doc = f.read()

        k = kml.KML()
        k.from_string(doc)

        # Retrieve Placemarks from KML
        features = list(k.features())
        f2 = list(features[0].features())

        for placemark in f2:
            if placemark.name.lower() == self.placemark_name.lower():
                if placemark.geometry.geom_type == 'Polygon':
                    polygon_geom = placemark.geometry
                    return Polygon(polygon_geom.exterior.coords)

        raise Exception(f"Placemark with name {self.placemark_name} containing a Polygon not found.")

    def point_within_rows_entry_polygons(self, gnss_position):
        point = Point(gnss_position)
        if self.rows_entry_polygons is not None:
            self.within_rows_entry_polygons = any(polygon.contains(point) for polygon in self.rows_entry_polygons)
            return self.within_rows_entry_polygons
        else:
            return None

    def is_within_inner_polygon(self, gnss_position):
        point = Point(gnss_position)
        self.within_inner_polygon = self.inner_polygon.contains(point)
        return self.within_inner_polygon

    def get_heading_bounds(self, heading, heading_threshold):
        self.lower_boundF, self.upper_boundF = RowDetector.get_heading_bounds_360(heading, heading_threshold)
        self.lower_boundR, self.upper_boundR = RowDetector.get_heading_bounds_360((heading + 180) % 360, heading_threshold)
        return self.lower_boundF, self.upper_boundF, self.lower_boundR, self.upper_boundR

    def update_dataframe_results(self, imu_timestamp, angular_velocity_x, gps_timestamp, longitude, latitude, ground_truth=None):

        update_values = {
            'imu_timestamp': imu_timestamp,
            'angular_velocity_x': angular_velocity_x,
            'gps_timestamp': gps_timestamp,
            'longitude': longitude,
            'latitude': latitude,
            'score': self.depth_score,
            'heading_180': self.heading_360R,
            'heading_360': self.heading_360F,
            'row_heading': self.row_heading_360F,
            'depth_ema': self.depth_ema,
            'ang_vel_ema': self.angular_velocity_x_ema,
            'within_row_depth': self.within_row_depth,
            'within_row_angular_velocity': self.within_row_angular_velocity,
            'within_row_heading': self.within_row_heading,
            'within_rows_entry_polygons': self.within_rows_entry_polygons,
            'row_state': self.row_state.value,
            'row_length': self.row_length,
            'pred_changed': self.pred_changed,
            'pred': self.row_pred,
            'row_number': self.row_number if self.row_pred else None}

        for column, value in update_values.items():
            self.df.loc[self.index, column] = value

        if ground_truth is not None:
            self.df.loc[self.index, 'ground_truth'] = ground_truth

    def generate_new_kml_file(self, output_dir):
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
        placemark1.outerboundaryis = list(self.polygon.exterior.coords)
        placemark1.style = style

        # Draw polygons with green color
        for i, polygon in enumerate(self.rows_entry_polygons):
            placemark = kml.newpolygon(name=f'Polygon{i + 2}')  # Start from 2 since 1 is already used
            placemark.outerboundaryis = list(polygon.exterior.coords)

            # Change the linestyle color of the polygon to green
            linestyle = simplekml.LineStyle(color=simplekml.Color.green, width=3)  # Green outline
            style = simplekml.Style()
            style.polystyle = polystyle
            style.linestyle = linestyle
            placemark.style = style

        # Create a separate LineString for each pair of points with the same 'pred' value.
        for i in range(len(self.df) - 1):
            row1 = self.df.iloc[i]
            row2 = self.df.iloc[i + 1]
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
        start_label = kml.newpoint(name='Start', coords=[(self.df.iloc[0]['longitude'], self.df.iloc[0]['latitude'])])
        end_label = kml.newpoint(name='End', coords=[(self.df.iloc[-1]['longitude'], self.df.iloc[-1]['latitude'])])

        # Save the KML document to a new file
        output_path = os.path.join(output_dir, 'new_file.kml')
        kml.save(output_path)
        print(f'KML file saved to {output_path}')

    def plot_sensors(self, title, save_dir=None):

        plt.figure(figsize=(55, 35))
        sns.set(font_scale=2)

        n_subplots = 6
        self._subplot_(n_subplots=n_subplots, i_subplot=1, column_name1="score", column_name2="depth_ema",
                  thresh1=self.DEPTH_THRESHOLD, thresh2=None, thresh3=None, title='Depth score')

        self._subplot_(n_subplots=n_subplots, i_subplot=2, column_name1="angular_velocity_x", column_name2="ang_vel_ema",
                  thresh1=self.ANGULAR_VELOCITY_THRESHOLD, thresh2=-self.ANGULAR_VELOCITY_THRESHOLD, thresh3=None,
                  title='angular_velocity_x (deg/sec)')

        self._subplot_(n_subplots=n_subplots, i_subplot=3, column_name1="heading_360", column_name2=None,
                  thresh1=self.lower_boundF , thresh2=self.upper_boundF, thresh3=self.lower_boundR,
                  thresh4=self.upper_boundR, thresh5=self.EXPECTED_HEADING, title='heading')

        self._subplot_(n_subplots=n_subplots, i_subplot=4, column_name1='within_rows_entry_polygons', column_name2=None,
                  thresh1=None, thresh2=None, thresh3=None, title=f"rows_entry_polygons")

        self._subplot_(n_subplots=n_subplots, i_subplot=5, column_name1='row_state', column_name2=None,
                  thresh1=None, thresh2=None, thresh3=None,
                  title=f"'Row_state. 0: 'not in row', 1: 'starting row', 2: 'middle of row', 3: 'end of row'")

        self._subplot_(n_subplots=n_subplots, i_subplot=6, column_name1="pred", column_name2=None,
                  thresh1=None, thresh2=None, thresh3=None, title=f"Prediction. 0: 'not in row', 1: 'starting row'")

        plt.suptitle(title)
        plt.tight_layout()

        if save_dir:
            output_path = os.path.join(save_dir, f"plot_{title}.png")
            plt.savefig(output_path)
            plt.close()
            print(f'saved plot to {output_path}')

        plt.show()

    def _subplot_(self, n_subplots, i_subplot, title, column_name1, column_name2=None, thresh1=None, thresh2=None,
                  thresh3=None, thresh4=None, thresh5=None):

        plt.subplot(n_subplots, 1, i_subplot)

        # draw the ground truth:
        if 'ground_truth' in self.df.columns:
            plt.fill_between(self.df.index, self.df[column_name1].min(), self.df[column_name1].max(), where=self.df['ground_truth'] == 1,
                             color='green', alpha=0.15)

        # draw the plots:
        graph = sns.lineplot(data=self.df, x=self.df.index, y=column_name1)
        if column_name2:
            sns.lineplot(data=self.df, x=self.df.index, y=column_name2)

        # draw thresholds:
        if thresh1:
            graph.axhline(thresh1, color='red', linewidth=2)
            if thresh2:
                graph.axhline(thresh2, color='red', linewidth=2)
                if thresh3:
                    graph.axhline(thresh3, color='red', linewidth=2)
                    if thresh4:
                        graph.axhline(thresh4, color='red', linewidth=2)
                        if thresh5:
                            graph.axhline(thresh5, color='blue', linewidth=2)

        plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(200))
        plt.xlim(0, self.df.index[-1])
        plt.grid(True)
        plt.title(title)

    @staticmethod
    def get_distance(lon1, lat1, lon2, lat2):
        """
        Calculate the Haversine distance in meters, between two points.
        """
        # Radius of the Earth in kilometers
        R = 6371.0

        # Convert degrees to radians
        lat1_rad, lon1_rad, lat2_rad, lon2_rad = radians(lat1), radians(lon1), radians(lat2), radians(lon2)

        # Differences
        delta_lat = lat2_rad - lat1_rad
        delta_lon = lon2_rad - lon1_rad

        # Haversine formula
        a = sin(delta_lat / 2) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(delta_lon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        # Distance
        distance_meters = R * c * 1000
        return distance_meters

    @staticmethod
    def get_point_at_distance(lat1, lon1, d, bearing, R=6371):
        """
        lat: initial latitude, in degrees
        lon: initial longitude, in degrees
        d: target distance from initial
        bearing: (true) heading in degrees
        R: optional radius of sphere, defaults to mean radius of earth

        Returns new lat/lon coordinate {d}km from initial, in degrees
        """
        d = d / 1000  # convert km to meters
        lat1 = radians(lat1)
        lon1 = radians(lon1)
        a = radians(bearing)
        lat2 = asin(sin(lat1) * cos(d / R) + cos(lat1) * sin(d / R) * cos(a))
        lon2 = lon1 + atan2(
            sin(a) * sin(d / R) * cos(lat1),
            cos(d / R) - sin(lat1) * sin(lat2))
        return (degrees(lat2), degrees(lon2))

    @staticmethod
    def get_inner_polygon(polygon, margins_meters):
        # Convert self.MARGINS_THRESHOLD from meters to degrees (assuming a flat Earth approximation)
        degrees_per_meter = 1 / 111000
        buffer_distance_deg = margins_meters * degrees_per_meter

        # Create inner polygon by buffering original polygon with the converted distance
        inner_polygon = polygon.buffer(-buffer_distance_deg)

        # If the buffering operation results in multiple polygons, we take the largest one
        if inner_polygon.type == 'MultiPolygon':
            inner_polygon = max(inner_polygon, key=lambda x: x.area)
        return inner_polygon

    @staticmethod
    def _heading_within_range(current_heading, lower_bound, upper_bound):

        # Check if the current heading falls within the expected heading range
        if lower_bound <= upper_bound:
            is_in_heading = lower_bound <= current_heading <= upper_bound
        else:
            # Handle the case where the expected heading range wraps around 360 degrees
            is_in_heading = current_heading >= lower_bound or current_heading <= upper_bound
        return is_in_heading

    @staticmethod
    def get_heading_bounds_360(expected_heading, threshold):
        # Normalize the headings to be between 0 and 360 degrees
        expected_heading = expected_heading % 360

        # Calculate the lower and upper bounds for the expected heading range
        lower_bound = (expected_heading - threshold) % 360
        upper_bound = (expected_heading + threshold) % 360

        return lower_bound, upper_bound

    @staticmethod
    def exponential_moving_average(x, last_ema, alpha=0.5):
        return alpha * x + (1 - alpha) * last_ema

    @staticmethod
    def draw_lines_text(depth_img, y_low, y_high, score, ground_truth=None):

        text1 = f'Score: {score}'
        depth_3d = cv2.cvtColor(depth_img.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        depth_3d = cv2.putText(img=depth_3d, text=text1, org=(80, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                               fontScale=2, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)

        if ground_truth is not None:
            depth_3d = cv2.putText(img=depth_3d, text=f'GT: {ground_truth}', org=(700, 100), fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                                   fontScale=2, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)

        depth_3d = cv2.line(depth_3d, pt1=(0, y_high), pt2=(depth_img.shape[1], y_high), color=(0, 255, 0), thickness=5)
        depth_3d = cv2.line(depth_3d, pt1=(0, y_low), pt2=(depth_img.shape[1], y_low), color=(0, 255, 0), thickness=5)
        return depth_3d

    @staticmethod
    def heading_within_range_FR(heading, lower_boundF, upper_boundF, lower_boundR, upper_boundR):
        within_headingF = RowDetector._heading_within_range(heading, lower_boundF, upper_boundF)
        within_headingR = RowDetector._heading_within_range(heading, lower_boundR, upper_boundR)
        within_heading = within_headingF or within_headingR   #todo check if return none if heading is none
        return within_heading

    @staticmethod
    def percent_far_pixels(depth_img, rgb_img=None, show_video=False,
                           depth_window_y_high=0.35, depth_window_y_low=0.65):

        # Rescale the image from the range of 0-8 ? to the range of 1-3.5
        depth_img = (depth_img / 255) * 8
        depth_img = np.clip(depth_img, 1, 3.5)
        depth_img = cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        # Invert the grayscale image
        depth_img = 255 - depth_img

        # todo: un-comment
        # todo: check if RGB or BGR
        # # shadow sky noise:
        # blue = rgb_img[:, :, 0].copy()
        # depth[blue > 240] = 0

        y_high = int(depth_img.shape[0] * depth_window_y_high)
        y_low = int(depth_img.shape[0] * depth_window_y_low)

        width = depth_img.shape[1]
        search_area = depth_img[y_high: y_low, :].copy()
        score = np.sum(search_area < 20) / ((y_low - y_high) * width) # % pixels blow threshold
        depth_score = round(score, 2)

        # show video:
        if show_video:
            depth_3d = RowDetector.draw_lines_text(depth_img, y_low, y_high, depth_score)
            img_merged = cv2.hconcat([rgb_img, depth_3d])
            cv2.imshow('Video', cv2.resize(img_merged, (int(img_merged.shape[0] / 4), int(img_merged.shape[1] / 4))))

        return depth_score

