import pandas as pd
from vision.line_detection.retrieve_sensors_data import plot_sensors
import os
import cv2
import simplekml
from vision.line_detection.rows_detector import RowDetector
from vision.line_detection.retrieve_sensors_data import get_sensors_data
from tqdm import tqdm


def count_rows(column):
    mask = column == 1
    # Group consecutive "1" periods (cumsum) to assign a unique label to each group
    groups = mask.ne(mask.shift()).cumsum()
    # Filter only the groups where the value is 1 and count them
    rows_count = groups[mask].nunique()
    return rows_count


def generate_new_kml_file(polygon1, polygons, df, output_dir):
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

    # Draw polygons with green color
    for i, polygon in enumerate(polygons):
        placemark = kml.newpolygon(name=f'Polygon{i + 2}')  # Start from 2 since 1 is already used
        placemark.outerboundaryis = list(polygon.exterior.coords)

        # Change the linestyle color of the polygon to green
        linestyle = simplekml.LineStyle(color=simplekml.Color.green, width=3)  # Green outline
        style = simplekml.Style()
        style.polystyle = polystyle
        style.linestyle = linestyle
        placemark.style = style

    # Sort the DataFrame by timestamp (replace 'timestamp' with the appropriate column)
    # df = df.sort_values('timestamp')

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




if __name__ == '__main__':

    # Get sensors data to csv file:
    PATH_ROW = r'/home/lihi/FruitSpec/Data/customers/EinVered/2023_05_21/VALENCI2'
    PATH_OUTPUT = r'/home/lihi/FruitSpec/Data/customers/EinVered/2023_05_21/VALENCI2/row_10/1'
    output_dir = os.path.join(PATH_OUTPUT, 'rows_detection')
    df = get_sensors_data(PATH_ROW, output_dir, GT = True, save=False)


    CSV_PATH = os.path.join(output_dir, 'sensors_data.csv')
    DEPTH_VIDEO_PATH = r'/home/lihi/FruitSpec/Data/customers/EinVered/2023_05_21/VALENCI2/row_10/1/DEPTH.mkv'
    RGB_VIDEO_PATH = r'/home/lihi/FruitSpec/Data/customers/EinVered/2023_05_21/VALENCI2/row_10/1/ZED.mkv'
    PATH_KML = r'/home/lihi/FruitSpec/Data/customers/EinVered/Blocks.kml'
    #EXPECTED_HEADING = None
    EXPECTED_HEADING = 100

    PATH_ROW = os.path.dirname(os.path.dirname(CSV_PATH))
    output_dir = os.path.join(PATH_ROW, 'rows_detection')
    output_name = "_".join(PATH_ROW.split('/')[-4:])

    df = pd.read_csv(CSV_PATH)
    print(f'Loaded {CSV_PATH}')

    # init rows detector:
    row_detector = RowDetector(expected_heading = EXPECTED_HEADING, path_kml = PATH_KML, placemark_name = 'VALENCI2')

    # load video:
    cap_rgb = cv2.VideoCapture(RGB_VIDEO_PATH)
    cap_depth = cv2.VideoCapture(DEPTH_VIDEO_PATH)
    print(f'Loaded {DEPTH_VIDEO_PATH}')
    print (f'Video_len:{int(cap_depth.get(cv2.CAP_PROP_FRAME_COUNT))}, df_len:{len(df)}')

    # load frames:
    for index, row in tqdm(df.iterrows(), total=df.shape[0]):

        # Get depth and RGB frames:
        ret1, depth_img = cap_depth.read()
        if not ret1:
            break
        ret2, rgb_img = cap_rgb.read()
        if not ret2:
            break

        #rotate images:
        depth_img = cv2.rotate(depth_img, cv2.ROTATE_90_CLOCKWISE)   #cv2.ROTATE_90_COUNTERCLOCKWISE
        rgb_img = cv2.rotate(rgb_img, cv2.ROTATE_90_CLOCKWISE)


        depth_img = depth_img[:, :, 0].copy()
        gt = row.GT if 'GT' in df.columns.values else None
        is_row, state_changed, df_results = row_detector.detect_row(depth_img=depth_img,
                                                                    rgb_img=rgb_img,
                                                                    angular_velocity_x = row.angular_velocity_x,
                                                                    longitude = row.longitude, latitude = row.latitude,
                                                                    imu_timestamp =row.timestamp,
                                                                    gps_timestamp=row.timestamp_gnss,
                                                                    ground_truth = gt)

        if cv2.waitKey(25) & 0xFF == ord('q'):    # Wait for 25 milliseconds and check if the user pressed 'q' to quit
            break


    # save dataframe:
    df_results.to_csv(os.path.join(output_dir, f'rows_data.csv'), index=False)


    # rows_in_GT = count_rows(column= df['GT'])
    #rows_in_Pred = count_rows(column=df['pred'])

    generate_new_kml_file(row_detector.polygon, row_detector.rows_entry_polygons, df_results, output_dir=output_dir)

    # plot sensors data:
    config = f'EX2_Pred:Enter_depth_and_heading__Exit_ang_vel_DEPTH_THRESH {row_detector.DEPTH_THRESHOLD}, DEPTH_EMA {row_detector.DEPTH_EMA_ALPHA}, ANG_VEL_THRESH {row_detector.ANGULAR_VELOCITY_THRESHOLD}, ANG_VEL_EMA {row_detector.ANGULAR_VELOCITY_EMA_ALPHA}, EXPECTED_HEADING {row_detector.EXPECTED_HEADING}, HEADING_THRESH {row_detector.HEADING_THRESHOLD}_, self.MARGINS_THRESHOLD {row_detector.MARGINS_THRESHOLD}'

    plot_sensors(df_results, config + output_name,
                 depth_threshold = row_detector.DEPTH_THRESHOLD,
                 angular_velocity_threshold = row_detector.ANGULAR_VELOCITY_THRESHOLD,
                 expected_heading = row_detector.EXPECTED_HEADING,
                 lower_heading_bound = row_detector.lower_bound,
                 upper_heading_bound= row_detector.upper_bound,
                 margins_threshold = row_detector.MARGINS_THRESHOLD,
                 save_dir = output_dir)

    print('Done!')
