import logging
import os
import threading
import time
import signal
from datetime import datetime
from builtins import staticmethod
from application.utils.settings import GPS_conf, conf, analysis_conf, data_conf
from application.utils.module_wrapper import ModulesEnum, Module, ModuleTransferAction
from application.Analysis import batcher
import jaized
import cv2
import numpy as np


class AcquisitionManager(Module):
    acquisition_start_event = threading.Event()
    jz_recorder, batcher = None, None
    fps = -1
    exposure_rgb, exposure_800, exposure_975 = -1, -1, -1
    output_dir = ""
    output_clahe_fsi, output_equalize_hist_fsi = False, False
    output_rgb, output_800, output_975, output_svo = False, False, False, False
    view, debug_mode = False, False
    pass_clahe_stream = False

    @staticmethod
    def init_module(sender, receiver, main_pid, module_name):
        super(AcquisitionManager, AcquisitionManager).init_module(sender, receiver, main_pid, module_name)
        signal.signal(signal.SIGTERM, AcquisitionManager.shutdown)
        signal.signal(signal.SIGUSR1, AcquisitionManager.receive_data)
        AcquisitionManager.set_acquisition_parameters()
        AcquisitionManager.jz_recorder = jaized.JaiZed()
        AcquisitionManager.connect_cameras()
        AcquisitionManager.batcher = batcher.Batcher(AcquisitionManager.jz_recorder)
        AcquisitionManager.batcher.prepare_batches()

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
            AcquisitionManager.output_975, AcquisitionManager.output_svo, AcquisitionManager.view,
            AcquisitionManager.pass_clahe_stream, AcquisitionManager.debug_mode
        )
        AcquisitionManager.batcher.start_acquisition()

    @staticmethod
    def stop_acquisition():
        AcquisitionManager.jz_recorder.stop_acquisition()
        AcquisitionManager.batcher.stop_acquisition()

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
        AcquisitionManager.output_equalize_hist_fsi = 'equalize_hist' in output_types or 'fsi' in output_types
        AcquisitionManager.output_rgb = 'rgb' in output_types
        AcquisitionManager.output_800 = '800' in output_types
        AcquisitionManager.output_975 = '975' in output_types
        AcquisitionManager.output_svo = 'svo' in output_types
        AcquisitionManager.view = False
        AcquisitionManager.pass_clahe_stream = True
        AcquisitionManager.debug_mode = True

        if not os.path.exists(AcquisitionManager.output_dir):
            os.makedirs(AcquisitionManager.output_dir)

    @staticmethod
    def receive_data(sig, frame):
        data, sender_module = AcquisitionManager.receiver.recv()
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

    @staticmethod
    def pop_frames():
        while not AcquisitionManager.shutdown_done_event.is_set():
            AcquisitionManager.acquisition_start_event.wait()
            jai_frame = AcquisitionManager.jz_recorder.pop_jai()
            zed_frame = AcquisitionManager.jz_recorder.pop_zed()
            # if jai_frame.frame_number % 50 == 0:
            #     cv2.destroyAllWindows()
            #     cv2.imshow("mat", jai_frame.frame)
            #     cv2.waitKey(1000)
