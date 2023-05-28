import pandas as pd
from vision.line_detection.retrieve_sensors_data import plot_sensors
import os
import numpy as np
import cv2
from shapely.geometry import Point, Polygon
from fastkml import kml
import simplekml
from enum import Enum



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
        self.MARGINS_THRESHOLD = 3
        self.polygon = self.parse_kml_file()
        self.EXPECTED_HEADING = expected_heading
        self.HEADING_THRESHOLD = 30
        self.DEPTH_WINDOW_Y_HIGH = 0.35
        self.DEPTH_WINDOW_Y_LOW = 0.65
        self.DEPTH_THRESHOLD = 0.5
        self.DEPTH_EMA_ALPHA = 0.02
        self.ANGULAR_VELOCITY_THRESHOLD = 8  #10
        self.ANGULAR_VELOCITY_EMA_ALPHA = 0.02
        self.CONSISTENCY_THRESHOLD = 3
        self.lower_bound, self.upper_bound = self.get_heading_bounds_180(self.EXPECTED_HEADING, self.HEADING_THRESHOLD)

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
        self.heading_360, self.heading_180 =  self.get_heading(longitude, latitude)
        if self.heading_360 and self.heading_180 is not None:   # The first time the heading is None
            self.within_row_heading = self.heading_within_range(self.heading_180, self.lower_bound, self.upper_bound)
        self.within_inner_polygon = self.is_within_inner_polygon((longitude, latitude))

    def detect_row(self,  angular_velocity_x, longitude , latitude, rgb_img = None, depth_img = None):

        self.percent_far_pixels(depth_img, rgb_img = rgb_img, show_video =False)
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


