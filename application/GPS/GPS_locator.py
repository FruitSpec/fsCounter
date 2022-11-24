import geojson
from shapely.geometry import Polygon, Point
from kml2geojson import convert
from application.utils.settings import GPS_conf


class GPSLocator:
    def __init__(self, file_path):
        self.polygons = {}
        # convert from kml to json
        data = str(convert(file_path))
        # make it valid json
        data = data.replace("\'", "\"")
        json_data = geojson.loads(data)[0]
        for feature in json_data['features']:
            properties = feature['properties']
            plot_code = properties['name']
            geometry = feature['geometry']
            long_lat_coords = geometry['coordinates'][0]
            # that is because KML contains coordinates in long-lat but shapely reads lat-long
            lat_long_coords = [coord[::-1] for coord in long_lat_coords]
            curr_plot = Polygon(lat_long_coords)
            self.polygons[plot_code] = curr_plot
            # distance in degrees

    def find_containing_polygon(self, lat, long):
        curr_loc = Point([lat, long])
        for plot_code, polygon in self.polygons.items():
            # distance_to_edge = polygon.exterior.distance(curr_loc) * settings.lat_long_distance_scale
            if polygon.contains(curr_loc):
                #  and distance_to_edge > settings.inside_polygon_threshold
                return plot_code
        return GPS_conf["global polygon"]
