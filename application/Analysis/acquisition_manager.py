import logging
import os
import threading
import time
import signal
from datetime import datetime
from builtins import staticmethod
from application.utils.settings import conf, analysis_conf, data_conf, consts
from application.utils.module_wrapper import ModulesEnum, Module, ModuleTransferAction
from application.utils import tools

from application.Analysis.analysis_manager import AnalysisManager

import jaized


class AcquisitionManager(Module):
    acquisition_start_event = threading.Event()
    jz_recorder, analyzer = None, None
    jai_connected, zed_connected, running = False, False, False
    health_check_lock = threading.Lock()
    acquisition_lock = threading.Lock()
    receive_data_thread = None
    fps = -1
    exposure_rgb, exposure_800, exposure_975 = -1, -1, -1
    plot, row, folder_index = None, None, None
    output_dir = ""
    output_clahe_fsi, output_equalize_hist_fsi = False, False
    output_rgb, output_800, output_975, output_svo = False, False, False, False
    output_zed_gray, output_zed_depth, output_zed_pc = False, False, False
    view, debug_mode = False, False
    transfer_data, pass_clahe_stream = False, False

    @staticmethod
    def init_module(in_qu, out_qu, main_pid, module_name, communication_queue, notify_on_death, death_action):
        super(AcquisitionManager, AcquisitionManager).init_module(in_qu, out_qu, main_pid, module_name,
                                                                  communication_queue, notify_on_death, death_action)
        signal.signal(signal.SIGTERM, AcquisitionManager.shutdown)
        signal.signal(signal.SIGUSR2, AcquisitionManager.shutdown)

        AcquisitionManager.jz_recorder = jaized.JaiZed()
        AcquisitionManager.analyzer = AnalysisManager(AcquisitionManager.jz_recorder, AcquisitionManager.send_data)

        AcquisitionManager.receive_data_thread = threading.Thread(target=AcquisitionManager.receive_data, daemon=True)
        AcquisitionManager.receive_data_thread.start()

        AcquisitionManager.connect_cameras()
        # AcquisitionManager.cameras_health_check()
        AcquisitionManager.analyzer.start_analysis()

        AcquisitionManager.receive_data_thread.join()

    @staticmethod
    def cameras_health_check():
        def health_check():
            while not AcquisitionManager.shutdown_event.wait(1):
                with AcquisitionManager.health_check_lock:
                    actual_jai_connected = AcquisitionManager.jz_recorder.jai_connected()
                    actual_zed_connected = AcquisitionManager.jz_recorder.zed_connected()
                    actual_running = AcquisitionManager.jz_recorder.is_running()
                    if AcquisitionManager.jai_connected and (not actual_jai_connected):
                        AcquisitionManager.jz_recorder.disconect_jai()
                        AcquisitionManager.jai_connected = AcquisitionManager.jz_recorder.connect_jai()
                    if AcquisitionManager.zed_connected and (not actual_zed_connected):
                        AcquisitionManager.jz_recorder.disconect_zed()
                        AcquisitionManager.zed_connected = AcquisitionManager.jz_recorder.connect_zed(AcquisitionManager.fps)
                    if AcquisitionManager.running and not actual_running:
                        AcquisitionManager.start_acquisition(from_healthcheck=True)
        t = threading.Thread(target=health_check, daemon=True)
        t.start()

    @staticmethod
    def connect_cameras():
        AcquisitionManager.fps = 15
        AcquisitionManager.debug_mode = True
        with AcquisitionManager.health_check_lock:
            cam_status = AcquisitionManager.jz_recorder.connect_cameras(AcquisitionManager.fps,
                                                                        AcquisitionManager.debug_mode)
            AcquisitionManager.jai_connected, AcquisitionManager.zed_connected = cam_status
            if conf.GUI:
                AcquisitionManager.send_data(ModuleTransferAction.GUI_SET_DEVICE_STATE, cam_status, ModulesEnum.GUI)
                time.sleep(1)
                AcquisitionManager.send_data(ModuleTransferAction.START_GPS, None, ModulesEnum.GPS)
            else:
                AcquisitionManager.send_data(ModuleTransferAction.START_GPS, None, ModulesEnum.GPS)

    @staticmethod
    def disconnect_cameras():
        try:
            AcquisitionManager.jz_recorder.disconnect_cameras()
        except:
            pass

    @staticmethod
    def start_acquisition(acquisition_parameters=None, from_healthcheck=False, from_gps=False):
        with AcquisitionManager.acquisition_lock:
            AcquisitionManager.set_acquisition_parameters(
                data=acquisition_parameters,
                index_only=from_healthcheck,
                from_gps=from_gps
            )
            running = AcquisitionManager.jz_recorder.start_acquisition(
                AcquisitionManager.fps, AcquisitionManager.exposure_rgb, AcquisitionManager.exposure_800,
                AcquisitionManager.exposure_975, AcquisitionManager.output_dir, AcquisitionManager.output_clahe_fsi,
                AcquisitionManager.output_equalize_hist_fsi, AcquisitionManager.output_rgb,
                AcquisitionManager.output_800, AcquisitionManager.output_975, AcquisitionManager.output_svo,
                AcquisitionManager.output_zed_gray, AcquisitionManager.output_zed_depth, AcquisitionManager.output_zed_pc,
                AcquisitionManager.view, AcquisitionManager.transfer_data, AcquisitionManager.pass_clahe_stream,
                AcquisitionManager.debug_mode
            )
            print("RUNNING")
            if from_healthcheck:
                AcquisitionManager.running = running
            else:
                with AcquisitionManager.health_check_lock:
                    AcquisitionManager.running = running
            AcquisitionManager.analyzer.start_acquisition()

    @staticmethod
    def stop_acquisition():
        with AcquisitionManager.acquisition_lock:
            AcquisitionManager.analyzer.stop_acquisition()
            AcquisitionManager.jz_recorder.stop_acquisition()
            with AcquisitionManager.health_check_lock:
                AcquisitionManager.running = False

    @staticmethod
    def get_row_number(row_name):
        try:
            return int(row_name.split('_')[-1])
        except:
            return 0

    @staticmethod
    def set_acquisition_parameters(data, index_only=False, from_gps=False):
        if index_only:
            row_path = os.path.dirname(AcquisitionManager.output_dir)
            AcquisitionManager.row = os.path.basename(row_path)
            AcquisitionManager.folder_index = tools.get_folder_index(row_path)

            AcquisitionManager.output_dir = os.path.join(row_path, str(AcquisitionManager.folder_index))
            AcquisitionManager.analyzer.set_output_dir(AcquisitionManager.output_dir)

            if not os.path.exists(AcquisitionManager.output_dir):
                os.makedirs(AcquisitionManager.output_dir)

        else:
            if from_gps:
                AcquisitionManager.plot = data
                plot_dir = os.path.join(data_conf.output_path, conf.customer_code, AcquisitionManager.plot)
                today = datetime.now().strftime("%d%m%y")
                today_dir = os.path.join(plot_dir, today)
                if os.path.exists(today_dir):
                    new_row_number = 1 + max([AcquisitionManager.get_row_number(f) for f in os.listdir(today_dir)],
                                             default=0)
                else:
                    new_row_number = 1
                AcquisitionManager.row = f"row_{new_row_number}"
                row_path = os.path.join(plot_dir, today, AcquisitionManager.row)
                AcquisitionManager.folder_index = "1"
                AcquisitionManager.output_dir = os.path.join(row_path, AcquisitionManager.folder_index)

                if analysis_conf.autonomous_easy_config:
                    weather = analysis_conf.default_weather
                    camera_data = analysis_conf.default_acquisition_parameters[weather]
                    output_types = analysis_conf.default_acquisition_parameters.output_types
                else:
                    camera_data = analysis_conf.custom_acquisition_parameters
                    output_types = camera_data["default output types"]
            else:
                today = datetime.now().strftime("%d%m%y")
                AcquisitionManager.plot = data["plot"]
                AcquisitionManager.row = f"row_{data['row']}"
                row_path = os.path.join(data["outputPath"], conf.customer_code, AcquisitionManager.plot, today,
                                        AcquisitionManager.row)
                AcquisitionManager.folder_index = tools.get_folder_index(row_path)

                AcquisitionManager.output_dir = os.path.join(row_path, str(AcquisitionManager.folder_index))

                if 'Default' in data['configType']:
                    weather = data['weather']
                    camera_data = analysis_conf.default_acquisition_parameters[weather]
                    output_types = analysis_conf.default_acquisition_parameters.output_types
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
            AcquisitionManager.output_zed_gray = 'zed_gray' in output_types
            AcquisitionManager.output_zed_depth = 'zed_depth' in output_types
            AcquisitionManager.output_zed_pc = 'zed_pc' in output_types
            AcquisitionManager.view = False
            AcquisitionManager.transfer_data = True
            AcquisitionManager.pass_clahe_stream = False
            AcquisitionManager.debug_mode = True

            AcquisitionManager.analyzer.set_output_dir(AcquisitionManager.output_dir)

            if not os.path.exists(AcquisitionManager.output_dir):
                os.makedirs(AcquisitionManager.output_dir)

    @staticmethod
    def receive_data():
        logging.info("ACQUISITION MODULE receive_data STARTED")
        print("ACQUISITION MODULE receive_data STARTED")
        while True:
            data, sender_module = AcquisitionManager.in_qu.get()
            action, data = data["action"], data["data"]
            if sender_module == ModulesEnum.GPS:
                if action == ModuleTransferAction.ENTER_PLOT and conf.autonomous_acquisition:
                    AcquisitionManager.plot = data
                    AcquisitionManager.start_acquisition(AcquisitionManager.plot, from_gps=True)
                    row_name = "/".join([
                        conf.customer_code, AcquisitionManager.plot,
                        AcquisitionManager.row, str(AcquisitionManager.folder_index)
                    ])
                    logging.info(f"START ACQUISITION FROM GPS - {row_name}/")
                    data = {
                        "plot": AcquisitionManager.plot,
                        "row": AcquisitionManager.get_row_number(AcquisitionManager.row),
                        "folder_index": AcquisitionManager.folder_index
                    }
                    AcquisitionManager.send_data(ModuleTransferAction.START_ACQUISITION, data, ModulesEnum.DataManager)
                elif action == ModuleTransferAction.EXIT_PLOT and conf.autonomous_acquisition:
                    logging.info("STOP ACQUISITION FROM GPS")
                    AcquisitionManager.stop_acquisition()
            elif sender_module == ModulesEnum.GUI:
                if action == ModuleTransferAction.START_ACQUISITION:
                    logging.info("START ACQUISITION FROM GUI")
                    AcquisitionManager.start_acquisition(acquisition_parameters=data)
                    row_path = os.path.dirname(AcquisitionManager.output_dir)
                    data["folder_index"] = tools.get_folder_index(row_path, get_next_index=False)
                    AcquisitionManager.send_data(ModuleTransferAction.START_ACQUISITION, data, ModulesEnum.DataManager)
                elif action == ModuleTransferAction.STOP_ACQUISITION:
                    AcquisitionManager.stop_acquisition()
                    logging.info("STOP ACQUISITION FROM GUI")
            if sender_module == ModulesEnum.Main:
                if action == ModuleTransferAction.MONITOR:
                    AcquisitionManager.send_data(ModuleTransferAction.MONITOR, None, ModulesEnum.Main)

    @staticmethod
    def shutdown(sig, frame):
        print("acquisition shutdown")
        if AcquisitionManager.zed_connected or AcquisitionManager.zed_connected:
            AcquisitionManager.disconnect_cameras()
        if AcquisitionManager.running:
            AcquisitionManager.send_data(ModuleTransferAction.ACQUISITION_CRASH, None, ModulesEnum.GPS)
            AcquisitionManager.send_data(ModuleTransferAction.ACQUISITION_CRASH, None, ModulesEnum.DataManager)
