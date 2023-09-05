from math import atan2, cos, radians, sin, sqrt

class DistanceGPS:

    def __init__(self):
        self.longitude_previous = None
        self.latitude_previous = None

    def get_distance(self, lon1, lat1):
        """
        Calculate the Haversine distance in meters, between two points.
        """
        if self.latitude_previous is None or self.longitude_previous is None:
            self.longitude_previous = lon1
            self.latitude_previous = lat1
            return 0

        # Radius of the Earth in kilometers
        R = 6371.0

        # Convert degrees to radians
        lat1_rad, lon1_rad, lat2_rad, lon2_rad = radians(lat1), radians(lon1), radians(self.latitude_previous), radians(
            self.longitude_previous)

        # Differences
        delta_lat = lat2_rad - lat1_rad
        delta_lon = lon2_rad - lon1_rad

        # Haversine formula
        a = sin(delta_lat / 2) ** 2 + cos(lat1_rad) * cos(lat2_rad) * sin(delta_lon / 2) ** 2
        c = 2 * atan2(sqrt(a), sqrt(1 - a))

        # Distance
        distance_meters = R * c * 1000

        self.longitude_previous = lon1
        self.latitude_previous = lat1

        return distance_meters

if __name__ == "__main__":

    lon1 = 34.937737416666664
    lat1 = 32.2621471

    lon2 = 34.93794123333333
    lat2 = 32.26218886666667

    # init distance detector:
    dist_detector = DistanceGPS()

    # calculate distance between two points:
    dist = dist_detector.get_distance(lon1, lat1)
    print (f'Distance: {dist} meters') # since there is only 1 point yet, the distance is 0

    dist = dist_detector.get_distance(lon2, lat2)
    print (f'Distance: {dist} meters')


    # accumulative_distance: 19.87736697659144 meters
    # Distance: 19.719186996980007 meters