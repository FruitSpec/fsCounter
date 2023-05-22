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
import sys
import cv2
import numpy as np


class AcquisitionManager(Module):
    acquisition_start_event = threading.Event()
    jz_recorder, analyzer = None, None
    jai_connected, zed_connected, running = False, False, False
    health_check_lock = threading.Lock()
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
        AcquisitionManager.cameras_health_check()
        AcquisitionManager.analyzer.start_analysis()

    @staticmethod
    def cameras_health_check():
        def health_check():
            while not AcquisitionManager.shutdown_event.wait(1):
                with AcquisitionManager.health_check_lock:
                    actual_jai_connected = AcquisitionManager.jz_recorder.jai_connected()
                    actual_zed_connected = AcquisitionManager.jz_recorder.zed_connected()
                    actual_running = AcquisitionManager.jz_recorder.is_running()
                    if AcquisitionManager.jai_connected and (not actual_jai_connected):
                        print("reviving jai after crash")
                        AcquisitionManager.jz_recorder.disconect_jai()
                        AcquisitionManager.jai_connected = AcquisitionManager.jz_recorder.connect_jai()
                    if AcquisitionManager.zed_connected and (not actual_zed_connected):
                        print("reviving zed after crash")
                        AcquisitionManager.jz_recorder.disconect_zed()
                        AcquisitionManager.zed_connected = AcquisitionManager.jz_recorder.connect_zed(AcquisitionManager.fps)
                    if AcquisitionManager.running and not actual_running:
                        AcquisitionManager.start_acquisition(from_healthcheck=True)
        t = threading.Thread(target=health_check, daemon=True)
        t.start()

    @staticmethod
    def connect_cameras():
        jai_connected, zed_connected = AcquisitionManager.jz_recorder.connect_cameras(AcquisitionManager.fps,
                                                                                      AcquisitionManager.debug_mode)
        AcquisitionManager.send_data(ModuleTransferAction.GUI_SET_DEVICE_STATE, (jai_connected, zed_connected),
                                     ModulesEnum.GUI)
        with AcquisitionManager.health_check_lock:
            AcquisitionManager.jai_connected, AcquisitionManager.zed_connected = jai_connected, zed_connected

    @staticmethod
    def start_acquisition(acquisition_parameters=None, from_healthcheck=False):
        AcquisitionManager.set_acquisition_parameters(data=acquisition_parameters, index_only=from_healthcheck)
        running = AcquisitionManager.jz_recorder.start_acquisition(
            AcquisitionManager.fps, AcquisitionManager.exposure_rgb, AcquisitionManager.exposure_800,
            AcquisitionManager.exposure_975, AcquisitionManager.output_dir, AcquisitionManager.output_clahe_fsi,
            AcquisitionManager.output_equalize_hist_fsi, AcquisitionManager.output_rgb, AcquisitionManager.output_800,
            AcquisitionManager.output_975, AcquisitionManager.output_svo, AcquisitionManager.output_zed_mkv,
            AcquisitionManager.view, AcquisitionManager.transfer_data, AcquisitionManager.pass_clahe_stream,
            AcquisitionManager.debug_mode
        )
        if from_healthcheck:
            AcquisitionManager.running = running
        else:
            with AcquisitionManager.health_check_lock:
                AcquisitionManager.running = running
        AcquisitionManager.analyzer.start_acquisition()

    @staticmethod
    def stop_acquisition():
        AcquisitionManager.jz_recorder.stop_acquisition()

        with AcquisitionManager.health_check_lock:
            AcquisitionManager.running = False
        AcquisitionManager.analyzer.stop_acquisition()

    @staticmethod
    def get_scan_index(row_path):
        try:
            row_dirs = os.listdir(row_path)
            path_indices = [int(f) for f in row_dirs if os.path.isdir(os.path.join(row_path, f)) and f.isdigit()]
            scan_index = 1 + max(path_indices, default=0)
        except FileNotFoundError:
            scan_index = 1
        return scan_index

    @staticmethod
    def set_acquisition_parameters(data=None, index_only=False):
        if index_only:
            row_path = os.path.dirname(AcquisitionManager.output_dir)
            scan_index = AcquisitionManager.get_scan_index(row_path)

            AcquisitionManager.output_dir = os.path.join(row_path, str(scan_index))
            AcquisitionManager.analyzer.set_output_dir(AcquisitionManager.output_dir)

            if not os.path.exists(AcquisitionManager.output_dir):
                os.makedirs(AcquisitionManager.output_dir)

        else:
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
                row_path = os.path.join(data["outputPath"], conf["customer code"], plot, today, row)
                scan_index = AcquisitionManager.get_scan_index(row_path)

                AcquisitionManager.output_dir = os.path.join(row_path, str(scan_index))

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
        if type(sender_module) is tuple:
            print("ACQ MAN: ", sender_module, " ,", data)
            return
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
                AcquisitionManager.send_data(ModuleTransferAction.START_ACQUISITION, data, ModulesEnum.DataManager)
            elif action == ModuleTransferAction.STOP_ACQUISITION:
                AcquisitionManager.stop_acquisition()
                logging.info("STOP ACQUISITION FROM GUI")
                AcquisitionManager.send_data(ModuleTransferAction.STOP_ACQUISITION, None, ModulesEnum.DataManager)

