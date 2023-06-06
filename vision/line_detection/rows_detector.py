import pandas as pd
from vision.line_detection.retrieve_sensors_data import plot_sensors
import os
import numpy as np
import cv2
from shapely.geometry import Point, Polygon
from fastkml import kml
import simplekml
from enum import Enum
from math import asin, atan2, cos, degrees, radians, sin
import matplotlib.pyplot as plt



class RowState(Enum):
    NOT_IN_ROW = 0
    STARTING_ROW = 1
    MIDDLE_OF_ROW = 2
    ENDING_ROW = 3

class RowDetector:

    def __init__(self, path_kml, placemark_name, expected_heading = None):

        # Constants:
        self.path_kml = path_kml
        self.placemark_name = placemark_name
        self.MARGINS_THRESHOLD = 6
        self.polygon = self.parse_kml_file()
        self.EXPECTED_HEADING = expected_heading
        self.rows_entry_polygons = self.get_rows_entry_polygons()
        self.HEADING_THRESHOLD = 30
        self.DEPTH_WINDOW_Y_HIGH = 0.35
        self.DEPTH_WINDOW_Y_LOW = 0.65
        self.DEPTH_THRESHOLD = 0.5
        self.DEPTH_EMA_ALPHA = 0.02
        self.ANGULAR_VELOCITY_THRESHOLD = 8  #10
        self.ANGULAR_VELOCITY_EMA_ALPHA = 0.02
        self.CONSISTENCY_THRESHOLD = 3
        self.lower_bound, self.upper_bound = None, None
        if self.EXPECTED_HEADING is not None:
            self.lower_bound, self.upper_bound = self.get_heading_bounds_180(self.EXPECTED_HEADING, self.HEADING_THRESHOLD) # todo: calculate after extracting heading

        # Init:
        self.inner_polygon = self.get_inner_polygon(self.polygon, self.MARGINS_THRESHOLD)
        self.depth_ema = 0.5
        self.angular_velocity_x_ema = 0
        self.heading_360, self.heading_180 = None, None
        self.consistency_counter = 0
        self.previous_longitude = None
        self.previous_latitude = None
        self.depth_score = None
        self.within_row_angular_velocity = None
        self.within_row_depth = None
        self.within_row_heading = None
        self.within_rows_entry_polygons = None
        self.row_state = RowState.NOT_IN_ROW
        self.row_pred = None
        self.pred_changed = False
        self.point_inner_polygon_in = None
        self.point_inner_polygon_out = None
        self.df = pd.DataFrame()
        self.index = 0

    def get_rows_entry_polygons(self):
        self.rows_entry_polygons = []
        # calculate heading for polygon edges:
        coords = list(self.polygon.exterior.coords)
        for i in range(len(coords) - 1):
            lon1, lat1 = coords[i]
            lon2, lat2 = coords[i + 1]

            heading_360, heading_180, _, _ = self.get_heading(lon2, lat2, lon1, lat1)
            lower_bound, upper_bound = self.get_heading_bounds_180(self.EXPECTED_HEADING, 25)
            within_heading = self.heading_within_range(heading_180, lower_bound, upper_bound)

            # if the heading is not within range, create a new polygon
            if not within_heading:
                p1_lat1, p1_lon1 = self.get_point_at_distance(lat1, lon1, self.MARGINS_THRESHOLD,
                                                              self.EXPECTED_HEADING)
                p1_lat2, p1_lon2 = self.get_point_at_distance(lat1, lon1, self.MARGINS_THRESHOLD,
                                                              self.EXPECTED_HEADING + 180)
                p2_lat1, p2_lon1 = self.get_point_at_distance(lat2, lon2, self.MARGINS_THRESHOLD,
                                                              self.EXPECTED_HEADING)
                p2_lat2, p2_lon2 = self.get_point_at_distance(lat2, lon2, self.MARGINS_THRESHOLD,
                                                              self.EXPECTED_HEADING + 180)

                # Call the function with your coordinates

                # generate new polygon from the points p1_lat1, p1_lon1, p1_lat2, p1_lon2,  p2_lat1, p2_lon1, p2_lat2, p2_lon2:
                new_polygon = Polygon([(p1_lon1, p1_lat1), (p1_lon2, p1_lat2), (p2_lon2, p2_lat2), (p2_lon1, p2_lat1)])
                self.rows_entry_polygons.append(new_polygon)
                self.plot_points(new_polygon, lat1, lon1, p1_lat1, p1_lon1, p1_lat2, p1_lon2, lat2, lon2, p2_lat1,
                                 p2_lon1, p2_lat2, p2_lon2)
        return self.rows_entry_polygons


    def plot_points(self, new_polygon, lat0, lon0, lat1, lon1, lat2, lon2, lat3, lon3, lat4, lon4, lat5, lon5):
        # Define the coordinates of the points
        lat_values = [lat0, lat1, lat2, lat3, lat4, lat5]
        lon_values = [lon0, lon1, lon2, lon3, lon4, lon5]

        # Create the plot
        plt.scatter(lon_values, lat_values)

        # Add labels for the points
        for i, txt in enumerate(['Point 0', 'Point 1', 'Point 2', 'Point 10', 'Point 11', 'Point 12']):
            plt.annotate(txt, (lon_values[i], lat_values[i]))

        # plot the polygon:
        x,y = new_polygon.exterior.xy
        plt.plot(x,y)
        plt.show()




    def get_point_at_distance(self, lat1, lon1, d, bearing, R=6371):
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

    def global_decision(self, longitude , latitude):
        # Reset state_changed to False at the start of each call
        self.pred_changed = False
        self.row_heading = None

        # State: Not_in_Row
        if self.row_state == RowState.NOT_IN_ROW:
            # If heading is unknown:
            if self.EXPECTED_HEADING is None:
                if self.within_row_depth and self.within_row_angular_velocity and not (self.within_rows_entry_polygons):
                    self.consistency_counter += 1
                    if self.consistency_counter >= self.CONSISTENCY_THRESHOLD:
                        self.row_state = RowState.STARTING_ROW
                        self.consistency_counter = 0
                        self.pred_changed = True
                        self.point_inner_polygon_in = (longitude , latitude)
                else:
                    self.consistency_counter = 0

            # If heading is known:
            else:

                if self.within_row_depth and self.within_row_heading:               # if depth + heading => enter row
                    self.row_state = RowState.STARTING_ROW
                    self.pred_changed = True
                    self.point_inner_polygon_in = (longitude, latitude)
                else:
                    self.consistency_counter = 0

        # State: Starting a Row
        elif self.row_state == RowState.STARTING_ROW:
            if not (self.within_rows_entry_polygons):
                self.consistency_counter += 1
                if self.consistency_counter >= self.CONSISTENCY_THRESHOLD:
                    self.row_state = RowState.MIDDLE_OF_ROW
                    self.consistency_counter = 0
            else:
                self.consistency_counter = 0

        # State: Middle of a Row
        elif self.row_state == RowState.MIDDLE_OF_ROW:
            if not(self.within_rows_entry_polygons):             # todo: add delay?
                self.consistency_counter = 0
            else:
                self.consistency_counter += 1
                if self.consistency_counter >= self.CONSISTENCY_THRESHOLD:
                    self.row_state = RowState.ENDING_ROW
                    self.consistency_counter = 0
                    self.point_inner_polygon_out = (longitude, latitude)


        # State: Ending a Row
        elif self.row_state == RowState.ENDING_ROW:
            if not(self.within_rows_entry_polygons):
                self.row_state = RowState.MIDDLE_OF_ROW
            else:
                if not self.within_row_angular_velocity:
                    self.consistency_counter += 1
                    if self.consistency_counter >= self.CONSISTENCY_THRESHOLD:
                        self.row_state = RowState.NOT_IN_ROW
                        self.consistency_counter = 0
                        self.pred_changed = True

                        # Calculate heading:
                        row_heading_360, self.row_heading, _, _ = self.get_heading(
                            self.point_inner_polygon_out[0], self.point_inner_polygon_out[1], self.point_inner_polygon_in[0], self.point_inner_polygon_in[1])
                        if self.EXPECTED_HEADING is None:
                            self.EXPECTED_HEADING = self.row_heading
                            self.lower_bound, self.upper_bound = self.get_heading_bounds_180(self.EXPECTED_HEADING,
                                                                                                 self.HEADING_THRESHOLD)  # todo: calculate after extracting heading
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
        self.heading_360, self.heading_180, self.previous_longitude, self.previous_latitude  =  self.get_heading(longitude, latitude, self.previous_longitude, self.previous_latitude )

        if (self.heading_360 and self.heading_180) is not None:   # The first time the heading is None
            if (self.lower_bound and self.upper_bound) is not None: # In the first row the bounds are None
                self.within_row_heading = self.heading_within_range(self.heading_180, self.lower_bound, self.upper_bound)  # will be None if heading is None
        self.within_rows_entry_polygons = self.point_within_rows_entry_polygons((longitude, latitude))

    def detect_row(self,  angular_velocity_x, longitude , latitude, imu_timestamp, gps_timestamp, ground_truth = None, rgb_img = None, depth_img = None):

        self.percent_far_pixels(depth_img, rgb_img = rgb_img, show_video =False)
        self.sensors_decision(angular_velocity_x, longitude , latitude)
        self.global_decision(longitude , latitude)

        self.row_pred = int(self.row_state != RowState.NOT_IN_ROW)
        self.update_dataframe_results(imu_timestamp, angular_velocity_x, gps_timestamp, longitude, latitude, ground_truth)
        self.index += 1
        return self.row_pred, self.pred_changed, self.df

    def get_heading(self, longitude_curr, latitude_curr, longitude_previous, latitude_previous):
        '''the heading calculation assumes that the GNSS data is provided in the WGS84 coordinate system or a
        coordinate system where the north direction aligns with the positive y-axis. '''

        if latitude_previous and longitude_previous:  # The first time the heading is None
            # Calculate the difference in latitude and longitude
            delta_lat = latitude_curr - latitude_previous
            delta_lon = longitude_curr - longitude_previous
            # Calculate the heading using atan2
            heading_rad = np.arctan2(delta_lon, delta_lat)
            # Convert the heading from radians to degrees
            heading_deg = np.degrees(heading_rad)
            # Adjust the heading to be relative to the north
            heading_360 = (heading_deg + 360) % 360
            heading_180 = (heading_deg + 360) % 180
            return heading_360, heading_180, longitude_curr, latitude_curr

        else:
            return None, None, longitude_curr, latitude_curr

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


    def point_within_rows_entry_polygons(self, gnss_position):
        point = Point(gnss_position)
        self.within_rows_entry_polygons = any(polygon.contains(point) for polygon in self.rows_entry_polygons)
        return self.within_rows_entry_polygons

    @staticmethod
    def heading_within_range(current_heading, lower_bound, upper_bound):

        # Check if the current heading falls within the expected heading range
        if lower_bound <= upper_bound:
            is_in_heading = lower_bound <= current_heading <= upper_bound
        else:
            # Handle the case where the expected heading range wraps around 360 degrees
            is_in_heading = current_heading >= lower_bound or current_heading <= upper_bound

        return is_in_heading

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

    def percent_far_pixels(self, depth_img, rgb_img, show_video = False):

        # Rescale the image from the range of 0-8 ? to the range of 1-3.5
        depth_img = (depth_img / 255) * 8
        depth_img = np.clip(depth_img, 1, 3.5)
        depth_img= cv2.normalize(depth_img, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

        # Invert the grayscale image
        depth_img = 255 - depth_img

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

        # show video:
        if show_video:
            depth_3d = self.draw_lines_text(depth_img, y_low, y_high, self.depth_score)
            img_merged = cv2.hconcat([rgb_img, depth_3d])
            cv2.imshow('Video', cv2.resize(img_merged, (int(img_merged.shape[0] / 4), int(img_merged.shape[1] / 4))))

    def draw_lines_text(self, depth_img, y_low, y_high, score, ground_truth=None):

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

    def update_dataframe_results(self, imu_timestamp, angular_velocity_x, gps_timestamp, longitude, latitude, ground_truth=None):

        update_values = {
            'imu_timestamp': imu_timestamp,
            'angular_velocity_x': angular_velocity_x,
            'gps_timestamp': gps_timestamp,
            'longitude': longitude,
            'latitude': latitude,
            'score': self.depth_score,
            'heading_180': self.heading_180,
            'heading_360': self.heading_360,
            'row_heading': self.row_heading,
            'depth_ema': self.depth_ema,
            'ang_vel_ema': self.angular_velocity_x_ema,
            'within_row_depth': self.within_row_depth,
            'within_row_angular_velocity': self.within_row_angular_velocity,
            'within_row_heading': self.within_row_heading,
            'within_rows_entry_polygons': self.within_rows_entry_polygons,
            'row_state': self.row_state.value,
            'pred_changed': self.pred_changed,
            'pred': self.row_pred}

        for column, value in update_values.items():
            self.df.loc[self.index, column] = value

        if ground_truth is not None:
            self.df.loc[self.index, 'ground_truth'] = ground_truth





