import signal
import threading
import traceback
from builtins import staticmethod
from botocore.config import Config

import boto3
import fscloudutils.exceptions

from application.utils.settings import GPS_conf, conf, data_conf, consts
from application.utils.module_wrapper import ModulesEnum, Module, ModuleTransferAction
import application.utils.tools as tools
from vision.pipelines.ops.line_detection.rows_detector import RowState, RowDetector

from fscloudutils.utils import NavParser
import time
from datetime import datetime, timedelta
import logging
import serial
from application.GPS.GPS_locator import GPSLocator
from application.utils.settings import set_logger
from application.GPS.led_settings import LedSettings, LedColor


class GPSSampler(Module):
    kml_flag = False
    locator = None
    sample_thread = None
    # row_detector = None
    start_sample_event = threading.Event()
    jaized_log_dict = dict()
    current_timestamp = datetime.now().strftime(data_conf.timestamp_format)
    current_lat, current_long = 0, 0
    previous_plot, current_plot = consts.global_polygon, consts.global_polygon
    s3_client = None
    analysis_ongoing = False
    last_step_in, last_step_out = None, None

    @staticmethod
    def init_module(in_qu, out_qu, main_pid, module_name, communication_queue, notify_on_death, death_action):
        super(GPSSampler, GPSSampler).init_module(in_qu, out_qu, main_pid, module_name, communication_queue,
                                                  notify_on_death, death_action)
        super(GPSSampler, GPSSampler).set_signals(GPSSampler.shutdown)

        GPSSampler.last_step_in, GPSSampler.last_step_out = datetime.now(), datetime.now()
        GPSSampler.init_jaized_log_dict()
        GPSSampler.s3_client = boto3.client('s3', config=Config(retries={"total_max_attempts": 1}))
        GPSSampler.get_kml(once=True)
        GPSSampler.set_locator()

        GPSSampler.sample_thread = threading.Thread(target=GPSSampler.sample_gps, daemon=True)
        GPSSampler.receive_data_thread = threading.Thread(target=GPSSampler.receive_data, daemon=True)

        GPSSampler.sample_thread.start()
        GPSSampler.receive_data_thread.start()

        GPSSampler.sample_thread.join()
        GPSSampler.receive_data_thread.join()

    @staticmethod
    def get_kml(once=False):
        while not GPSSampler.kml_flag:
            try:
                kml_aws_path = tools.s3_path_join(conf.customer_code, GPS_conf.s3_kml_file_name)
                GPSSampler.s3_client.download_file(GPS_conf.kml_bucket_name, kml_aws_path, GPS_conf.kml_path)
                GPSSampler.kml_flag = True
                logging.info("LATEST KML FILE RETRIEVED")
            except Exception:
                logging.info("KML FILE NOT RETRIEVED")
                if once:
                    break
                time.sleep(30)

    @staticmethod
    def set_locator():
        t = threading.Thread(target=GPSSampler.get_kml, daemon=True)
        t.start()
        time.sleep(1)
        while not (GPSSampler.locator or GPSSampler.shutdown_event.is_set()):
            try:
                GPSSampler.locator = GPSLocator(GPS_conf.kml_path)
                logging.info("LOCATOR INITIALIZED")
            except Exception:
                logging.error("LOCATOR COULD NOT BE INITIALIZED - RETRYING IN 30 SECONDS...")
                time.sleep(30)

    @staticmethod
    def init_jaized_log_dict():
        GPSSampler.jaized_log_dict = {
            consts.JAI_frame_number: [],
            consts.JAI_timestamp: [],
            consts.ZED_frame_number: [],
            consts.ZED_timestamp: [],
            consts.IMU_angular_velocity: [],
            consts.IMU_linear_acceleration: [],
            consts.row_state: [],
            consts.GPS_timestamp: [],
            consts.GPS_latitude: [],
            consts.GPS_longitude: [],
            consts.GPS_plot: []
        }

    @staticmethod
    def receive_data():
        while True:
            data, sender_module = GPSSampler.in_qu.get()
            action, data = data["action"], data["data"]
            if sender_module == ModulesEnum.Acquisition:
                if action == ModuleTransferAction.START_GPS:
                    GPSSampler.start_sample_event.set()
                if action == ModuleTransferAction.JAIZED_TIMESTAMPS:

                    row_state = GPSSampler.get_row_state(
                        angular_velocity_x=data[consts.IMU_angular_velocity][0],
                        lat=GPSSampler.current_lat,
                        long=GPSSampler.current_long,
                        imu_timestamp=data[consts.ZED_timestamp],
                        gps_timestamp=GPSSampler.current_timestamp,
                        depth_score=data[consts.depth_score]
                    )

                    angular_velocity = data[consts.IMU_angular_velocity]
                    linear_acceleration = data[consts.IMU_linear_acceleration]

                    GPSSampler.jaized_log_dict[consts.JAI_frame_number].append(data[consts.JAI_frame_number])
                    GPSSampler.jaized_log_dict[consts.JAI_timestamp].append(data[consts.JAI_timestamp])
                    GPSSampler.jaized_log_dict[consts.ZED_frame_number].append(data[consts.ZED_frame_number])
                    GPSSampler.jaized_log_dict[consts.ZED_timestamp].append(data[consts.ZED_timestamp])
                    GPSSampler.jaized_log_dict[consts.IMU_angular_velocity].append(angular_velocity)
                    GPSSampler.jaized_log_dict[consts.IMU_linear_acceleration].append(linear_acceleration)
                    GPSSampler.jaized_log_dict[consts.row_state].append(row_state)
                    GPSSampler.jaized_log_dict[consts.GPS_timestamp].append(GPSSampler.current_timestamp)
                    GPSSampler.jaized_log_dict[consts.GPS_latitude].append(GPSSampler.current_lat)
                    GPSSampler.jaized_log_dict[consts.GPS_longitude].append(GPSSampler.current_long)
                    GPSSampler.jaized_log_dict[consts.GPS_plot].append(GPSSampler.current_plot)

                if action == ModuleTransferAction.ACQUISITION_CRASH:
                    GPSSampler.start_sample_event.clear()
                    time.sleep(1)
                    GPSSampler.previous_plot = consts.global_polygon
                    LedSettings.turn_on(LedColor.RED)
            if sender_module == ModulesEnum.Analysis:
                if action == ModuleTransferAction.ANALYSIS_ONGOING:
                    GPSSampler.analysis_ongoing = True
                if action == ModuleTransferAction.ANALYSIS_DONE:
                    GPSSampler.analysis_ongoing = False
            if sender_module == ModulesEnum.Main:
                if action == ModuleTransferAction.MONITOR:
                    GPSSampler.send_data(ModuleTransferAction.MONITOR, None, ModulesEnum.Main)
                elif action == ModuleTransferAction.SET_LOGGER:
                    set_logger()

    @staticmethod
    def sample_gps():
        logging.info("START")
        parser = NavParser("", is_file=False)
        ser = None
        while not GPSSampler.shutdown_event.is_set():
            try:
                ser = serial.Serial(GPS_conf.GPS_device_name, timeout=1, )
                ser.flushOutput()
                ser.flushInput()
                logging.info(f"SERIAL PORT INIT - SUCCESS")
                break
            except (serial.SerialException, TimeoutError) as e:
                logging.warning(f"SERIAL PORT ERROR - RETRYING IN 5...")
                time.sleep(5)
            except Exception:
                logging.exception(f"UNKNOWN SERIAL PORT ERROR - RETRYING IN 5...")
                time.sleep(5)
        err_count = 0
        sample_count = 0
        GPSSampler.gps_data = []
        while not GPSSampler.shutdown_event.is_set():
            is_start_sample = GPSSampler.start_sample_event.wait(10)
            if not is_start_sample:
                LedSettings.turn_on(LedColor.RED)
                continue
            data = ""
            while ser.in_waiting > 0:
                data += ser.readline().decode('utf-8')
            if not data:
                continue
            timestamp = datetime.now().strftime(data_conf.timestamp_format)
            try:
                parser.read_string(data)
                point = parser.get_most_recent_point()
                lat, long = point.get_lat(), point.get_long()
                GPSSampler.previous_plot = GPSSampler.current_plot
                plot = GPSSampler.locator.find_containing_polygon(lat=lat, long=long)

                GPSSampler.current_plot = plot
                GPSSampler.current_lat = lat
                GPSSampler.current_long = long
                GPSSampler.current_timestamp = timestamp

                sample_count += 1

                if sample_count % 20 == 0 and GPSSampler.jaized_log_dict[consts.JAI_frame_number]:
                    GPSSampler.send_data(
                        action=ModuleTransferAction.JAIZED_TIMESTAMPS,
                        data=GPSSampler.jaized_log_dict,
                        receiver=ModulesEnum.DataManager
                    )

                    GPSSampler.init_jaized_log_dict()

                GPSSampler.gps_data.append(
                    {
                        consts.GPS_timestamp: timestamp,
                        consts.GPS_latitude: lat,
                        consts.GPS_longitude: long,
                        consts.GPS_plot: GPSSampler.current_plot
                    }
                )

                if sample_count % 20 == 0 and GPSSampler.gps_data:
                    GPSSampler.send_data(ModuleTransferAction.NAV, GPSSampler.gps_data, ModulesEnum.DataManager)
                    GPSSampler.gps_data = []

                if GPSSampler.current_plot != GPSSampler.previous_plot:  # Switched to another block
                    # stepped into new block
                    if GPSSampler.previous_plot == consts.global_polygon:
                        state_changed = GPSSampler.step_in()
                    else:
                        state_changed = GPSSampler.step_out()
                    if not state_changed:
                        GPSSampler.current_plot = GPSSampler.previous_plot

                # check if in global
                if GPSSampler.current_plot == consts.global_polygon:
                    if GPSSampler.analysis_ongoing:
                        LedSettings.start_blinking(LedColor.ORANGE, LedColor.BLINK_TRANSPARENT)
                    else:
                        LedSettings.turn_on(LedColor.ORANGE)
                else:
                    LedSettings.turn_on(LedColor.GREEN)

                err_count = 0

            except fscloudutils.exceptions.InputError:
                print(data)
            except ValueError as e:
                timestamp = datetime.now().strftime(data_conf.timestamp_format)
                GPSSampler.gps_data.append(
                    {
                        consts.GPS_timestamp: timestamp,
                        consts.GPS_latitude: None,
                        consts.GPS_longitude: None,
                        consts.GPS_plot: GPSSampler.current_plot
                    }
                )
                sample_count += 1
                err_count += 1
                if err_count in {1, 10, 30} or err_count % 60 == 0:
                    logging.error(f"{err_count} SECONDS WITH NO GPS (CONSECUTIVE)")
                # release the last detected block into Global if it is over 300 sec without GPS
                if err_count > 300 and GPSSampler.current_plot != consts.global_polygon:
                    GPSSampler.current_plot = consts.global_polygon
                    GPSSampler.step_out()
                LedSettings.turn_on(LedColor.RED)
                traceback.print_exc()
            except Exception:
                logging.exception("SAMPLE UNEXPECTED EXCEPTION")
                traceback.print_exc()
                LedSettings.turn_on(LedColor.RED)

        try:
            ser.close()
        except AttributeError:
            pass

        logging.info("END")
        LedSettings.turn_on(LedColor.RED)
        GPSSampler.shutdown_done_event.set()

    @staticmethod
    def step_in():
        if GPSSampler.last_step_out + timedelta(seconds=3) < datetime.now():
            # GPSSampler.row_detector = RowDetector(GPS_conf.kml_path, GPSSampler.current_plot)
            print(f"STEP IN {GPSSampler.current_plot}")
            logging.info(f"STEP IN {GPSSampler.current_plot}")
            GPSSampler.last_step_in = datetime.now()
            GPSSampler.send_data(ModuleTransferAction.ENTER_PLOT, GPSSampler.current_plot, ModulesEnum.Acquisition)
            return True
        else:
            print(f"DID NOT STEP IN {GPSSampler.current_plot}")
            logging.info(f"DID NOT STEP IN {GPSSampler.current_plot}")
            return False

    @staticmethod
    def step_out():
        if GPSSampler.last_step_in + timedelta(seconds=3) < datetime.now():
            print(f"STEP OUT {GPSSampler.previous_plot}")
            logging.info(f"STEP OUT {GPSSampler.previous_plot}")

            GPSSampler.last_step_out = datetime.now()

            if GPSSampler.jaized_log_dict[consts.JAI_frame_number]:
                GPSSampler.send_data(
                    action=ModuleTransferAction.JAIZED_TIMESTAMPS,
                    data=GPSSampler.jaized_log_dict,
                    receiver=ModulesEnum.DataManager
                )

                GPSSampler.init_jaized_log_dict()
                time.sleep(0.5)
            GPSSampler.send_data(ModuleTransferAction.EXIT_PLOT, None, ModulesEnum.Acquisition)
            return True
        else:
            return False

    @staticmethod
    def get_row_state(angular_velocity_x, lat, long, imu_timestamp, gps_timestamp, depth_score):
        # GPSSampler.row_detector.detect_row(
        #     angular_velocity_x=angular_velocity_x,
        #     latitude=lat,
        #     longitude=long,
        #     imu_timestamp=imu_timestamp,
        #     gps_timestamp=gps_timestamp,
        #     depth_score=depth_score
        # )
        # return GPSSampler.row_detector.row_state
        return -1
