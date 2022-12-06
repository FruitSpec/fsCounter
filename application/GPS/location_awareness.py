import signal
import threading
from builtins import staticmethod
import os
from application.utils.settings import GPS_conf
from application.utils.module_wrapper import ModulesEnum, Module
from fscloudutils.utils import NavParser
import time
from datetime import datetime
import logging
import serial
from application.GPS.GPS_locator import GPSLocator
from application.GPS.led_settings import LedSettings, LedColor


class GPSSampler(Module):
    global_polygon = GPS_conf["global polygon"]
    shutdown_event = threading.Event()
    is_latest = False
    
    @staticmethod
    def init_module(sender, receiver, main_pid):
        GPSSampler.sender = sender
        GPSSampler.receiver = receiver
        GPSSampler.main_pid = main_pid
        signal.signal(signal.SIGTERM, GPSSampler.shutdown)
        signal.signal(signal.SIGUSR1, GPSSampler.receive_data)
        GPSSampler.previous_plot, GPSSampler.plot_code = GPS_conf["global polygon"], GPS_conf["global polygon"]
        GPSSampler.locator, GPSSampler.is_latest = None, False
        GPSSampler.try_set_locator()

    @staticmethod
    def try_set_locator():
        if not GPSSampler.is_latest:
            try:
                GPSSampler.locator = GPSLocator(GPS_conf["kml path"])
                GPSSampler.is_latest = True
            except Exception:
                pass

    @staticmethod
    def receive_data(sig, frame):
        data, sender_module = GPSSampler.receiver.recv()

    @staticmethod
    def sample_gps():
        def sample():
            logging.info("START GPS")
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
                GPSSampler.try_set_locator()
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
                    GPSSampler.previous_plot = GPSSampler.plot_code
                    GPSSampler.plot_code = GPSSampler.locator.find_containing_polygon(lat=lat, long=long)

                    if GPSSampler.plot_code == GPSSampler.global_polygon:
                        LedSettings.turn_on(LedColor.ORANGE)
                    else:
                        LedSettings.turn_on(LedColor.GREEN)

                    gps_data = (GPS_conf["customer code"], scan_date, lat, long, GPSSampler.plot_code, timestamp)
                    GPSSampler.send_data(gps_data, ModulesEnum.DataManager)

                    if GPSSampler.plot_code != GPSSampler.previous_plot:  # Switched to another block

                        logging.info(f"BLOCK SWITCH: FROM {GPSSampler.previous_plot} TO {GPSSampler.plot_code}")

                        if GPSSampler.previous_plot == GPSSampler.global_polygon:  # stepped into new block
                            GPSSampler.block_switch()
                            GPSSampler.step_into()

                        elif GPSSampler.plot_code == GPSSampler.global_polygon:  # stepped out from block
                            GPSSampler.step_out()

                        else:  # moved from one block to another
                            GPSSampler.block_switch()
                            GPSSampler.step_out()
                            time.sleep(0.2)
                            # enable GPSSampler.streamer.capture to change file name for svo and writing samples
                            # to the correct block
                            GPSSampler.step_into()
                    err_count = 0

                except ValueError as e:
                    err_count += 1
                    if err_count in {1, 10, 30} or err_count % 60 == 0:
                        logging.error(f"{err_count} SECONDS WITH NO GPS (CONSECUTIVE)")
                    # release the last detected block into Global if it is over 300 sec without GPS
                    if err_count > 300 and GPSSampler.plot_code != GPSSampler.global_polygon:
                        GPSSampler.plot_code = GPSSampler.global_polygon
                        GPSSampler.step_out(turn_orange=False)
                    LedSettings.turn_on(LedColor.RED)
                except Exception:
                    logging.exception("GPS SAMPLE UNEXPECTED EXCEPTION")
                    LedSettings.turn_on(LedColor.RED)

            ser.close()
            logging.info("END GPS")
            LedSettings.turn_off()

        t = threading.Thread(target=sample, daemon=True)
        t.start()

    @staticmethod
    def shutdown(sig, frame):
        GPSSampler.shutdown_event.set()
