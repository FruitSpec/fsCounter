import logging
import os
import threading
import time
import signal
from datetime import datetime
from builtins import staticmethod
from application.utils.settings import GPS_conf, conf, analysis_conf, data_conf
from application.utils.module_wrapper import ModulesEnum, Module, ModuleTransferAction
from application.Analysis.analysis_manager import AnalysisManager
import jaized
import cv2
import numpy as np


class AcquisitionManager(Module):
    acquisition_start_event = threading.Event()
    jz_recorder, analyzer = None, None
    fps = -1
    exposure_rgb, exposure_800, exposure_975 = -1, -1, -1
    output_dir = ""
    output_clahe_fsi, output_equalize_hist_fsi = False, False
    output_rgb, output_800, output_975, output_svo, output_zed_mkv = False, False, False, False, False
    view, debug_mode = False, False
    transfer_data, pass_clahe_stream = False, False

    @staticmethod
    def init_module(qu, main_pid, module_name):
        super(AcquisitionManager, AcquisitionManager).init_module(qu, main_pid, module_name)
        signal.signal(signal.SIGTERM, AcquisitionManager.shutdown)
        signal.signal(signal.SIGUSR1, AcquisitionManager.receive_data)
        AcquisitionManager.jz_recorder = jaized.JaiZed()
        AcquisitionManager.analyzer = AnalysisManager(AcquisitionManager.jz_recorder)
        AcquisitionManager.set_acquisition_parameters()
        AcquisitionManager.connect_cameras()
        AcquisitionManager.analyzer.start_analysis()

    @staticmethod
    def connect_cameras():
        jai_connected, zed_connected = AcquisitionManager.jz_recorder.connect_cameras(AcquisitionManager.fps,
                                                                                      AcquisitionManager.debug_mode)
        AcquisitionManager.send_data(ModuleTransferAction.GUI_SET_DEVICE_STATE, (jai_connected, zed_connected),
                                     ModulesEnum.GUI)

    @staticmethod
    def start_acquisition(acquisition_parameters=None):
        AcquisitionManager.set_acquisition_parameters(acquisition_parameters)
        AcquisitionManager.jz_recorder.start_acquisition(
            AcquisitionManager.fps, AcquisitionManager.exposure_rgb, AcquisitionManager.exposure_800,
            AcquisitionManager.exposure_975, AcquisitionManager.output_dir, AcquisitionManager.output_clahe_fsi,
            AcquisitionManager.output_equalize_hist_fsi, AcquisitionManager.output_rgb, AcquisitionManager.output_800,
            AcquisitionManager.output_975, AcquisitionManager.output_svo, AcquisitionManager.output_zed_mkv,
            AcquisitionManager.view, AcquisitionManager.transfer_data, AcquisitionManager.pass_clahe_stream,
            AcquisitionManager.debug_mode
        )
        AcquisitionManager.analyzer.start_acquisition()

    @staticmethod
    def stop_acquisition():
        AcquisitionManager.jz_recorder.stop_acquisition()
        AcquisitionManager.analyzer.stop_acquisition()

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
            plot = data["plot"]
            row = f"row_{data['row']}"
            AcquisitionManager.output_dir = os.path.join(data["outputPath"], conf["customer code"], plot, today, row)
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
        AcquisitionManager.output_clahe_fsi = 'clahe' in output_types or 'fsi' in output_types
        AcquisitionManager.output_equalize_hist_fsi = 'equalize hist' in output_types or 'fsi' in output_types
        AcquisitionManager.output_rgb = 'rgb' in output_types
        AcquisitionManager.output_800 = '800' in output_types
        AcquisitionManager.output_975 = '975' in output_types
        AcquisitionManager.output_svo = 'svo' in output_types
        AcquisitionManager.output_zed_mkv = 'zed' in output_types
        AcquisitionManager.view = False
        AcquisitionManager.transfer_data = True
        AcquisitionManager.pass_clahe_stream = False
        AcquisitionManager.debug_mode = True

        AcquisitionManager.analyzer.set_output_dir(AcquisitionManager.output_dir)

        if not os.path.exists(AcquisitionManager.output_dir):
            os.makedirs(AcquisitionManager.output_dir)

    @staticmethod
    def receive_data(sig, frame):
        data, sender_module = AcquisitionManager.qu.get()
        action, data = data["action"], data["data"]
        global_polygon = GPS_conf["global polygon"]
        if sender_module == ModulesEnum.GPS:
            if data != global_polygon:
                logging.info("START ACQUISITION FROM GPS")
                AcquisitionManager.start_acquisition()
            else:
                AcquisitionManager.stop_acquisition()
                logging.info("STOP ACQUISITION FROM GPS")
        elif sender_module == ModulesEnum.GUI:
            if action == ModuleTransferAction.START_ACQUISITION:
                logging.info("START ACQUISITION FROM GUI")
                AcquisitionManager.start_acquisition(acquisition_parameters=data)
            elif action == ModuleTransferAction.STOP_ACQUISITION:
                AcquisitionManager.stop_acquisition()
                logging.info("STOP ACQUISITION FROM GUI")
