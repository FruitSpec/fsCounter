import logging
import os
import threading
from datetime import datetime
from builtins import staticmethod
from application.utils.settings import GPS_conf, conf, analysis_conf, data_conf
from application.utils.module_wrapper import ModulesEnum, Module
import jaized
import signal
import cv2
import numpy as np


class AcquisitionManager(Module):
    is_started = False
    jz_recorder = None
    fps = -1
    exposure_rgb, exposure_800, exposure_975 = -1, -1, -1
    output_dir = ""
    output_fsi, output_rgb, output_800, output_975, output_svo = False, False, False, False, False
    view, use_clahe_stretch, debug_mode = False, False, False

    @staticmethod
    def init_module(sender, receiver, main_pid, module_name):
        super(AcquisitionManager, AcquisitionManager).init_module(sender, receiver, main_pid, module_name)
        signal.signal(signal.SIGTERM, AcquisitionManager.shutdown)
        signal.signal(signal.SIGUSR1, AcquisitionManager.receive_data)
        AcquisitionManager.set_acquisition_parameters()
        AcquisitionManager.jz_recorder = jaized.JaiZed()
        AcquisitionManager.connect_cameras()
        AcquisitionManager.pop_frames()

    @staticmethod
    def connect_cameras():
        jai_connected, zed_connected = AcquisitionManager.jz_recorder.connect_cameras(AcquisitionManager.fps,
                                                                                      AcquisitionManager.debug_mode)
        AcquisitionManager.send_data("is connected", (jai_connected, zed_connected), ModulesEnum.GUI)

    @staticmethod
    def set_acquisition_parameters(data=None):
        if not data:
            AcquisitionManager.output_dir = os.path.join(data_conf['output path'], 'jaized_app_temp')

            if analysis_conf["autonomous easy config"]:
                weather = analysis_conf["default weather"]
                camera_data = analysis_conf["default acquisition parameters"][weather]
                output_types = analysis_conf["default acquisition parameters"]["output types"]
            else:
                camera_data = analysis_conf["custom acquisition parameters"]
                output_types = camera_data["default output types"]
        else:
            today = datetime.now().strftime("%d%m%y")
            AcquisitionManager.output_dir = os.path.join(data["outputPath"], data["plot"], today, f"row_{data['row']}")
            if 'Default' in data['configType']:
                weather = data['weather']
                camera_data = analysis_conf["default acquisition parameters"][weather]
                output_types = analysis_conf["default acquisition parameters"]["output types"]
            else:
                camera_data = data["Cameras"]
                output_types = camera_data["outputTypes"]

        output_types = [ot.lower() for ot in output_types]

        AcquisitionManager.fps = int(camera_data['FPS'])
        AcquisitionManager.exposure_rgb = int(camera_data['IntegrationTimeRGB'])
        AcquisitionManager.exposure_800 = int(camera_data['IntegrationTime800'])
        AcquisitionManager.exposure_975 = int(camera_data['IntegrationTime975'])
        AcquisitionManager.output_fsi = 'fsi' in output_types
        AcquisitionManager.output_rgb = 'rgb' in output_types
        AcquisitionManager.output_800 = '800' in output_types
        AcquisitionManager.output_975 = '975' in output_types
        AcquisitionManager.output_svo = 'svo' in output_types
        AcquisitionManager.view = False
        AcquisitionManager.use_clahe_stretch = False
        AcquisitionManager.debug_mode = True

        if not os.path.exists(AcquisitionManager.output_dir):
            os.makedirs(AcquisitionManager.output_dir)

    @staticmethod
    def receive_data(sig, frame):
        data, sender_module = AcquisitionManager.receiver.recv()
        action, data = data["action"], data["data"]
        if sender_module == ModulesEnum.GPS:
            if data != GPS_conf["global polygon"]:
                logging.info("START RECORDING FROM GPS")
                AcquisitionManager.set_acquisition_parameters()
                AcquisitionManager.jz_recorder.\
                    start_acquisition(AcquisitionManager.fps, AcquisitionManager.exposure_rgb,
                                      AcquisitionManager.exposure_800, AcquisitionManager.exposure_975,
                                      AcquisitionManager.output_dir, AcquisitionManager.output_fsi,
                                      AcquisitionManager.output_rgb, AcquisitionManager.output_800,
                                      AcquisitionManager.output_975, AcquisitionManager.output_svo,
                                      AcquisitionManager.view, AcquisitionManager.use_clahe_stretch,
                                      AcquisitionManager.debug_mode)
                AcquisitionManager.is_started = True
            else:
                logging.info("STOP RECORDING FROM GPS")
                AcquisitionManager.jz_recorder.stop_acquisition()
                AcquisitionManager.is_started = False
        elif sender_module == ModulesEnum.GUI:
            if action == "start":
                logging.info("START RECORDING FROM GUI")
                AcquisitionManager.set_acquisition_parameters(data)
                AcquisitionManager.jz_recorder.start_acquisition(
                    AcquisitionManager.fps, AcquisitionManager.exposure_rgb, AcquisitionManager.exposure_800,
                    AcquisitionManager.exposure_975, AcquisitionManager.output_dir, AcquisitionManager.output_fsi,
                    AcquisitionManager.output_rgb, AcquisitionManager.output_800, AcquisitionManager.output_975,
                    AcquisitionManager.output_svo, AcquisitionManager.view, AcquisitionManager.use_clahe_stretch,
                    AcquisitionManager.debug_mode)
                AcquisitionManager.is_started = True
            elif action == "stop":
                AcquisitionManager.jz_recorder.stop_acquisition()
                AcquisitionManager.is_started = False
                logging.info("STOP RECORDING FROM GUI")

    @staticmethod
    def pop_frames():
        while not AcquisitionManager.shutdown_done_event.is_set():
            if AcquisitionManager.is_started:
                A = AcquisitionManager.jz_recorder.pop()
            else:
                continue
