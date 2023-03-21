import signal
import threading
from builtins import staticmethod

import boto3

from application.utils.settings import GPS_conf, conf
from application.utils.module_wrapper import ModulesEnum, Module
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
    global_polygon = GPS_conf["global polygon"]
    shutdown_event = threading.Event()
    shutdown_done_event = threading.Event()
    locator = None
    is_latest = False
    previous_plot, current_plot = GPS_conf["global polygon"], GPS_conf["global polygon"]

    @staticmethod
    def init_module(sender, receiver, main_pid):
        GPSSampler.get_kml(once=True)
        GPSSampler.sender = sender
        GPSSampler.receiver = receiver
        GPSSampler.main_pid = main_pid
        signal.signal(signal.SIGTERM, GPSSampler.shutdown)
        signal.signal(signal.SIGUSR1, GPSSampler.receive_data)
        GPSSampler.set_locator()
        GPSSampler.sample_gps()  # non-blocking call

    @staticmethod
    def get_kml(once=False):
        while not GPSSampler.kml_flag:
            try:
                s3_client = boto3.client('s3')
                kml_aws_path = tools.s3_path_join(conf['customer code'], GPS_conf['s3 kml file name'])
                s3_client.download_file(GPS_conf["kml bucket name"], kml_aws_path, GPS_conf["kml path"])
                GPSSampler.kml_flag = True
                logging.info("GPS - LATEST KML FILE RETRIEVED")
            except Exception:
                if once:
                    break
                time.sleep(30)

    @staticmethod
    def set_locator():
        t = threading.Thread(target=GPSSampler.get_kml(), daemon=True)
        t.start()
        time.sleep(1)
        while not (GPSSampler.is_latest or GPSSampler.shutdown_event.is_set()):
            try:
                GPSSampler.locator = GPSLocator(GPS_conf["kml path"])
                GPSSampler.is_latest = True
            except Exception:
                time.sleep(30)

    @staticmethod
    def receive_data(sig, frame):
        data, sender_module = GPSSampler.receiver.recv()

    @staticmethod
    def sample_gps():
        def sample():
            logging.info("GPS - START")
            parser = NavParser("", is_file=False)
            ser = None
            while not GPSSampler.shutdown_event.is_set():
                try:
                    ser = serial.Serial("/dev/ttyUSB1", timeout=1, )
                    ser.flushOutput()
                    ser.flushInput()
                    logging.info(f"GPS - SERIAL PORT INIT - SUCCESS")
                    break
                except serial.SerialException:
                    logging.info(f"GPS - SERIAL PORT ERROR - RETRYING IN 5...")
                    time.sleep(5)
            err_count = 0
            while not GPSSampler.shutdown_event.is_set():
                data = ""
                while ser.in_waiting > 0:
                    data += ser.readline().decode('utf-8')
                if not data:
                    continue
                scan_date = datetime.utcnow().strftime("%d%m%y")
                timestamp = datetime.utcnow().strftime("%H:%M:%S")
                try:
                    parser.read_string(data)
                    point = parser.get_most_recent_point()
                    lat, long = point.get_lat(), point.get_long()
                    GPSSampler.previous_plot = GPSSampler.current_plot
                    GPSSampler.current_plot = GPSSampler.locator.find_containing_polygon(lat=lat, long=long)

                    if GPSSampler.current_plot == GPSSampler.global_polygon:
                        LedSettings.turn_on(LedColor.ORANGE)
                    else:
                        LedSettings.turn_on(LedColor.GREEN)

                    gps_data = (GPS_conf["customer code"], scan_date, lat, long, GPSSampler.current_plot, timestamp)
                    GPSSampler.send_data(gps_data, ModulesEnum.DataManager)

                    if GPSSampler.current_plot != GPSSampler.previous_plot:  # Switched to another block
                        # stepped into new block
                        if GPSSampler.previous_plot == GPSSampler.global_polygon:
                            GPSSampler.step_in()

                        # stepped out from block
                        elif GPSSampler.current_plot == GPSSampler.global_polygon:
                            GPSSampler.step_out()

                        # moved from one block to another
                        else:
                            GPSSampler.step_out()
                            time.sleep(0.2)
                            GPSSampler.step_in()
                    err_count = 0

                except ValueError as e:
                    err_count += 1
                    if err_count in {1, 10, 30} or err_count % 60 == 0:
                        logging.error(f"GPS - {err_count} SECONDS WITH NO GPS (CONSECUTIVE)")
                    # release the last detected block into Global if it is over 300 sec without GPS
                    if err_count > 300 and GPSSampler.current_plot != GPSSampler.global_polygon:
                        GPSSampler.current_plot = GPSSampler.global_polygon
                        GPSSampler.step_out(turn_orange=False)
                    LedSettings.turn_on(LedColor.RED)
                except Exception:
                    logging.exception("GPS - SAMPLE UNEXPECTED EXCEPTION")
                    LedSettings.turn_on(LedColor.RED)

            ser.close()
            logging.info("GPS - END")
            LedSettings.turn_off()
            GPSSampler.shutdown_done_event.set()

        t = threading.Thread(target=sample, daemon=True)
        t.start()

    # @staticmethod
    # def block_switch():
    #     path = os.path.join(settings.output_path, self.save_path)
    #     GPSSampler.update_file_index(path)  # update file index
    #     if not os.path.exists(path):  # create path if needed
    #         Path(path).mkdir(parents=True, exist_ok=True)
    #     os.kill(settings.server_pid, signal.SIGUSR1)
    #     globals.path_sender.send((path, self.file_index))
    #     logging.info(f"CLIENT MANUAL BLOCK SWITCH - PATH: {path} FILE INDEX: {self.file_index}")
        
    @staticmethod
    def step_in():
        logging.info(f"GPS - STEP IN {GPSSampler.current_plot}")
        GPSSampler.send_data(GPSSampler.current_plot, ModulesEnum.DataManager)

    @staticmethod
    def step_out():
        logging.info(f"GPS - STEP OUT {GPSSampler.previous_plot}")
