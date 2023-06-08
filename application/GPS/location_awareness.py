import signal
import threading
import traceback
from builtins import staticmethod

import boto3
import fscloudutils.exceptions

from application.utils.settings import GPS_conf, conf
from application.utils.module_wrapper import ModulesEnum, Module, ModuleTransferAction
import application.utils.tools as tools
from fscloudutils.utils import NavParser
import time
from datetime import datetime
import logging
import serial
from application.GPS.GPS_locator import GPSLocator
from application.GPS.led_settings import LedSettings, LedColor


class GPSSampler(Module):
    kml_flag = False
    locator = None
    sample_thread = None
    previous_plot, current_plot = GPS_conf.global_polygon, GPS_conf.global_polygon

    @staticmethod
    def init_module(in_qu, out_qu, main_pid, module_name, communication_queue):
        GPSSampler.get_kml(once=True)
        super(GPSSampler, GPSSampler).init_module(in_qu, out_qu, main_pid, module_name, communication_queue)
        super(GPSSampler, GPSSampler).set_signals(GPSSampler.shutdown, GPSSampler.receive_data)

        GPSSampler.set_locator()
        GPSSampler.sample_thread = threading.Thread(target=GPSSampler.sample_gps, daemon=True)
        GPSSampler.sample_thread.start()
        GPSSampler.sample_thread.join()

    @staticmethod
    def get_kml(once=False):
        while not GPSSampler.kml_flag:
            try:
                s3_client = boto3.client('s3')
                kml_aws_path = tools.s3_path_join(conf.customer_code, GPS_conf.s3_kml_file_name)
                s3_client.download_file(GPS_conf.kml_bucket_name, kml_aws_path, GPS_conf.kml_path)
                GPSSampler.kml_flag = True
                logging.info("LATEST KML FILE RETRIEVED")
            except Exception:
                logging.info("KML FILE NOT RETRIEVED")
                if once:
                    break
                time.sleep(30)

    @staticmethod
    def set_locator():
        t = threading.Thread(target=GPSSampler.get_kml  , daemon=True)
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
    def receive_data(sig, frame):
        data, sender_module = GPSSampler.in_qu.get()
        action, data = data["action"], data["data"]

    @staticmethod
    def sample_gps():
        logging.info("START")
        parser = NavParser("", is_file=False)
        ser = None
        while not GPSSampler.shutdown_event.is_set():
            try:
                ser = serial.Serial("/dev/ttyUSB1", timeout=1, )
                ser.flushOutput()
                ser.flushInput()
                logging.info(f"SERIAL PORT INIT - SUCCESS")
                break
            except serial.SerialException:
                logging.info(f"SERIAL PORT ERROR - RETRYING IN 5...")
                time.sleep(5)
        err_count = 0
        sample_count = 0
        switch_count = 0
        allow_step_out = False
        gps_data = []
        while not GPSSampler.shutdown_event.is_set():
            data = ""
            while ser.in_waiting > 0:
                data += ser.readline().decode('utf-8')
            if not data:
                continue
            timestamp = datetime.now().strftime("%H:%M:%S.%f")
            try:
                parser.read_string(data)
                point = parser.get_most_recent_point()
                lat, long = point.get_lat(), point.get_long()
                GPSSampler.previous_plot = GPSSampler.current_plot
                GPSSampler.current_plot = GPSSampler.locator.find_containing_polygon(lat=lat, long=long)

                if GPSSampler.current_plot == GPS_conf.global_polygon:
                    LedSettings.turn_on(LedColor.ORANGE)
                else:
                    LedSettings.turn_on(LedColor.GREEN)
                sample_count += 1
                gps_data.append(
                    {
                        "timestamp": timestamp,
                        "latitude": lat,
                        "longitude": long,
                        "plot": GPSSampler.current_plot
                    }
                )

                if sample_count % 30 == 0:
                    GPSSampler.send_data(ModuleTransferAction.NAV, gps_data, ModulesEnum.DataManager)
                    gps_data = []

                if GPSSampler.current_plot != GPSSampler.previous_plot:  # Switched to another block
                    # stepped into new block
                    if GPSSampler.previous_plot == GPS_conf.global_polygon:
                        GPSSampler.step_in()

                    # stepped out from block
                    elif GPSSampler.current_plot == GPS_conf.global_polygon:
                        GPSSampler.step_out()

                    # moved from one block to another
                    else:
                        GPSSampler.step_out()
                        time.sleep(3)
                        GPSSampler.step_in()
                err_count = 0
            except fscloudutils.exceptions.InputError:
                print(data)
            except ValueError as e:
                err_count += 1
                if err_count in {1, 10, 30} or err_count % 60 == 0:
                    logging.error(f"{err_count} SECONDS WITH NO GPS (CONSECUTIVE)")
                # release the last detected block into Global if it is over 300 sec without GPS
                if err_count > 300 and GPSSampler.current_plot != GPS_conf.global_polygon:
                    GPSSampler.current_plot = GPS_conf.global_polygon
                    GPSSampler.step_out()
                LedSettings.turn_on(LedColor.RED)
            except Exception:
                logging.exception("SAMPLE UNEXPECTED EXCEPTION")
                traceback.print_exc()
                LedSettings.turn_on(LedColor.RED)

        print("GPS ABOUT TO END. SHUTDOWN EVENT STATUS: ", GPSSampler.shutdown_event.is_set())
        ser.close()
        logging.info("END")
        LedSettings.turn_off()
        GPSSampler.shutdown_done_event.set()

    @staticmethod
    def step_in():
        print(f"STEP IN {GPSSampler.current_plot}")
        logging.info(f"STEP IN {GPSSampler.current_plot}")
        GPSSampler.send_data(ModuleTransferAction.ENTER_PLOT, GPSSampler.current_plot, ModulesEnum.Acquisition)

    @staticmethod
    def step_out():
        print(f"STEP OUT {GPSSampler.previous_plot}")
        logging.info(f"STEP OUT {GPSSampler.previous_plot}")
        GPSSampler.send_data(ModuleTransferAction.EXIT_PLOT, None, ModulesEnum.Acquisition)
