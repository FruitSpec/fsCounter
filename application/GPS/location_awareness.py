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
    is_in_plot = False
    s3_client = None
    analysis_ongoing = False
    last_step_in, last_step_out = None, None

    @staticmethod
    def init_module(in_qu, out_qu, main_pid, module_name, communication_queue, notify_on_death, death_action):
        super(GPSSampler, GPSSampler).init_module(in_qu, out_qu, main_pid, module_name, communication_queue,
                                                  notify_on_death, death_action)
        super(GPSSampler, GPSSampler).set_signals(GPSSampler.shutdown)

        LedSettings.turn_on(LedColor.RED)
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
        while not (GPSSampler.kml_flag or GPSSampler.shutdown_event.is_set()):
            try:
                kml_aws_path = tools.s3_path_join(conf.customer_code, GPS_conf.s3_kml_file_name)
                GPSSampler.s3_client.download_file(GPS_conf.kml_bucket_name, kml_aws_path, GPS_conf.kml_path)
                GPSSampler.kml_flag = True
                tools.log("LATEST KML FILE RETRIEVED")
            except Exception:
                tools.log("KML FILE NOT RETRIEVED")
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
                tools.log("LOCATOR INITIALIZED")
            except Exception:
                tools.log("LOCATOR COULD NOT BE INITIALIZED - RETRYING IN 30 SECONDS...", logging.ERROR, exc_info=True)
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
            consts.is_recording: [],
            consts.GPS_timestamp: [],
            consts.GPS_latitude: [],
            consts.GPS_longitude: [],
            consts.GPS_plot: []
        }

    @staticmethod
    def receive_data():
        while not GPSSampler.shutdown_event.is_set():
            data, sender_module = GPSSampler.in_qu.get()
            action, data = data["action"], data["data"]
            if sender_module == ModulesEnum.Acquisition:
                if action == ModuleTransferAction.START_GPS:
                    GPSSampler.start_sample_event.set()
                if action == ModuleTransferAction.JAIZED_TIMESTAMPS:
                    is_recording, jaized_data = data
                    if is_recording or conf.debug.constant_jaized:
                        row_state = GPSSampler.get_row_state(
                            angular_velocity_x=jaized_data[consts.IMU_angular_velocity][0],
                            lat=GPSSampler.current_lat,
                            long=GPSSampler.current_long,
                            imu_timestamp=jaized_data[consts.ZED_timestamp],
                            gps_timestamp=GPSSampler.current_timestamp,
                            depth_score=jaized_data[consts.depth_score]
                        )

                        angular_velocity = jaized_data[consts.IMU_angular_velocity]
                        linear_acceleration = jaized_data[consts.IMU_linear_acceleration]

                        GPSSampler.jaized_log_dict[consts.JAI_frame_number].append(jaized_data[consts.JAI_frame_number])
                        GPSSampler.jaized_log_dict[consts.JAI_timestamp].append(jaized_data[consts.JAI_timestamp])
                        GPSSampler.jaized_log_dict[consts.ZED_frame_number].append(jaized_data[consts.ZED_frame_number])
                        GPSSampler.jaized_log_dict[consts.ZED_timestamp].append(jaized_data[consts.ZED_timestamp])
                        GPSSampler.jaized_log_dict[consts.IMU_angular_velocity].append(angular_velocity)
                        GPSSampler.jaized_log_dict[consts.IMU_linear_acceleration].append(linear_acceleration)
                        GPSSampler.jaized_log_dict[consts.row_state].append(row_state)
                        GPSSampler.jaized_log_dict[consts.is_recording].append(is_recording)
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
                    GPSSampler.send_data(
                        action=ModuleTransferAction.MONITOR,
                        data=None,
                        receiver=ModulesEnum.Main,
                        log_option=tools.LogOptions.NONE
                    )
                elif action == ModuleTransferAction.SET_LOGGER:
                    set_logger()

    @staticmethod
    def sample_gps():

        def init_serial_port():
            while not GPSSampler.shutdown_event.is_set():
                try:
                    _ser = serial.Serial(GPS_conf.GPS_device_name, timeout=1, )
                    _ser.flushOutput()
                    _ser.flushInput()
                    tools.log(f"SERIAL PORT INIT - SUCCESS")
                    return _ser
                except (serial.SerialException, TimeoutError) as e:
                    tools.log(f"SERIAL PORT ERROR - RETRYING IN 5...", logging.WARNING)
                    time.sleep(5)
                except Exception:
                    tools.log(f"UNKNOWN SERIAL PORT ERROR - RETRYING IN 5...", logging.ERROR, exc_info=True)
                    time.sleep(5)

        tools.log("START")
        err_count = sample_count = 0
        parser = NavParser("", is_file=False)
        ser = init_serial_port()
        GPSSampler.gps_data = []

        while not GPSSampler.shutdown_event.is_set():
            is_start_sample = GPSSampler.start_sample_event.wait(10)
            if not is_start_sample:
                LedSettings.turn_on(LedColor.RED)
                continue

            # read NMEA data from the serial port
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

                # send the jaized_timestamps to the DataManager
                if sample_count % GPS_conf.sample_count_threshold == 0 \
                        and GPSSampler.jaized_log_dict[consts.JAI_frame_number]:
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

                if sample_count % GPS_conf.sample_count_threshold == 0 and GPSSampler.gps_data:
                    GPSSampler.send_data(
                        action=ModuleTransferAction.NAV,
                        data=GPSSampler.gps_data,
                        receiver=ModulesEnum.DataManager,
                        log_option=tools.LogOptions.LOG
                    )
                    GPSSampler.gps_data = []

                if GPSSampler.current_plot != GPSSampler.previous_plot:  # Switched to another block
                    # stepped into new block
                    if GPSSampler.previous_plot == consts.global_polygon:
                        GPSSampler.is_in_plot = GPSSampler.step_in()
                    else:
                        GPSSampler.is_in_plot = GPSSampler.step_out()
                        if GPSSampler.is_in_plot:
                            GPSSampler.current_plot = consts.global_polygon

                    if not GPSSampler.is_in_plot:
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
                GPSSampler.send_data(ModuleTransferAction.RESTART_APP, None, ModulesEnum.Main)
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
                    tools.log(f"{err_count} SECONDS WITH NO GPS (CONSECUTIVE)", logging.ERROR)
                # release the last detected block into Global if it is over no_gps_in_plot_limit seconds without GPS
                if err_count > GPS_conf.no_gps_in_plot_limit and GPSSampler.current_plot != consts.global_polygon:
                    GPSSampler.current_plot = consts.global_polygon
                    GPSSampler.step_out()
                LedSettings.turn_on(LedColor.RED)
            except Exception:
                tools.log("SAMPLE UNEXPECTED EXCEPTION", logging.ERROR, exc_info=True)
                LedSettings.turn_on(LedColor.RED)

        try:
            ser.close()
        except AttributeError:
            pass

        tools.log("END")
        LedSettings.turn_on(LedColor.RED)

    @staticmethod
    def step_in():
        if GPSSampler.last_step_out + timedelta(seconds=GPS_conf.minimal_time_in_state) < datetime.now():
            # GPSSampler.row_detector = RowDetector(GPS_conf.kml_path, GPSSampler.current_plot)
            tools.log(f"STEP IN {GPSSampler.current_plot}")
            GPSSampler.last_step_in = datetime.now()
            GPSSampler.send_data(ModuleTransferAction.ENTER_PLOT, GPSSampler.current_plot, ModulesEnum.Acquisition)
            return True
        else:
            tools.log(f"DID NOT STEP IN {GPSSampler.current_plot}")
            return False

    @staticmethod
    def step_out():
        if GPSSampler.last_step_in + timedelta(seconds=GPS_conf.minimal_time_in_state) < datetime.now():
            tools.log(f"STEP OUT {GPSSampler.previous_plot}")

            GPSSampler.last_step_out = datetime.now()

            if GPSSampler.jaized_log_dict[consts.JAI_frame_number]:
                GPSSampler.send_data(
                    action=ModuleTransferAction.JAIZED_TIMESTAMPS,
                    data=GPSSampler.jaized_log_dict,
                    receiver=ModulesEnum.DataManager
                )

                GPSSampler.init_jaized_log_dict()

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
