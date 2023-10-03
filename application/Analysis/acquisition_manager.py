import logging
import os
import threading
import time
import signal
import traceback
from datetime import datetime
from builtins import staticmethod
from application.utils.settings import conf, analysis_conf, data_conf, consts
from application.utils.module_wrapper import ModulesEnum, Module, ModuleTransferAction
from application.utils import tools
from application.utils.settings import set_logger

from application.Analysis.analysis_manager import AnalysisManager

import jaized


class AcquisitionManager(Module):
    acquisition_start_event = threading.Event()
    receive_data_t = None
    jz_recorder, analyzer = None, None
    jai_connected, zed_connected, running, recording = False, False, False, False

    exposure_rgb, exposure_800, exposure_975 = -1, -1, -1
    plot, row, folder_index = None, None, None
    output_dir = ""
    output_clahe_fsi, output_equalize_hist_fsi = False, False
    output_rgb, output_800, output_975, output_svo = False, False, False, False
    output_zed_gray, output_zed_depth, output_zed_pc = False, False, False
    debug_mode = False
    transfer_data, pass_clahe_stream = False, False
    alc_false_areas, alc_true_areas = [], []

    @staticmethod
    def init_module(in_qu, out_qu, main_pid, module_name, communication_queue, notify_on_death, death_action):
        super(AcquisitionManager, AcquisitionManager).init_module(in_qu, out_qu, main_pid, module_name,
                                                                  communication_queue, notify_on_death, death_action)
        signal.signal(signal.SIGTERM, AcquisitionManager.shutdown)
        signal.signal(signal.SIGUSR2, AcquisitionManager.shutdown)

        AcquisitionManager.jz_recorder = jaized.JaiZed()
        AcquisitionManager.analyzer = AnalysisManager(
            AcquisitionManager.jz_recorder,
            AcquisitionManager.send_data,
            AcquisitionManager.shutdown_event
        )

        AcquisitionManager.receive_data_t = threading.Thread(target=AcquisitionManager.receive_data, daemon=True)
        AcquisitionManager.receive_data_t.start()

        AcquisitionManager.connect_cameras()
        AcquisitionManager.start_acquisition()

        AcquisitionManager.analyzer.start_analysis()

        AcquisitionManager.receive_data_t.join()

    @staticmethod
    def connect_cameras():
        AcquisitionManager.debug_mode = True
        cam_status = AcquisitionManager.jz_recorder.connect_cameras(AcquisitionManager.debug_mode)
        AcquisitionManager.jai_connected, AcquisitionManager.zed_connected = cam_status
        if conf.GUI:
            AcquisitionManager.send_data(ModuleTransferAction.GUI_SET_DEVICE_STATE, cam_status, ModulesEnum.GUI)
            time.sleep(1)
            AcquisitionManager.send_data(ModuleTransferAction.START_GPS, None, ModulesEnum.GPS)
        else:
            AcquisitionManager.send_data(ModuleTransferAction.START_GPS, None, ModulesEnum.GPS)

    @staticmethod
    def start_acquisition():
        AcquisitionManager.set_acquisition_parameters()

        running = AcquisitionManager.jz_recorder.start_acquisition(
            AcquisitionManager.exposure_rgb, AcquisitionManager.exposure_800,
            AcquisitionManager.exposure_975, AcquisitionManager.transfer_data,
            AcquisitionManager.alc_true_areas, AcquisitionManager.alc_false_areas
        )

        AcquisitionManager.running = running
        AcquisitionManager.analyzer.start_acquisition()

    @staticmethod
    def start_recording(recording_parameters, from_gps):
        AcquisitionManager.set_recording_parameters(recording_parameters=recording_parameters, from_gps=from_gps)
        AcquisitionManager.analyzer.start_recording()

    @staticmethod
    def stop_recording():

        pass

    @staticmethod
    def stop_acquisition():
        pass

    @staticmethod
    def disconnect_cameras():
        try:
            tools.log("DISCONNECTING CAMERAS...")
            AcquisitionManager.jz_recorder.disconnect_cameras()
        except:
            traceback.print_exc()

    @staticmethod
    def stop_acquisition():
        AcquisitionManager.analyzer.stop_acquisition()
        AcquisitionManager.jz_recorder.stop_acquisition()
        AcquisitionManager.running = False

    @staticmethod
    def get_row_number(row_name):
        try:
            return int(row_name.split('_')[-1])
        except:
            return 0

    @staticmethod
    def set_acquisition_parameters():
        AcquisitionManager.exposure_rgb = int(analysis_conf.acquisition_parameters['IntegrationTimeRGB'])
        AcquisitionManager.exposure_800 = int(analysis_conf.acquisition_parameters['IntegrationTime800'])
        AcquisitionManager.exposure_975 = int(analysis_conf.acquisition_parameters['IntegrationTime975'])
        AcquisitionManager.transfer_data = True

    @staticmethod
    def set_recording_parameters(recording_parameters, from_gps=False):
        if from_gps:
            AcquisitionManager.plot = recording_parameters
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

            acquisition_parameters = analysis_conf.acquisition_parameters
            output_types = acquisition_parameters.output_types
        else:
            today = datetime.now().strftime("%d%m%y")
            AcquisitionManager.plot = recording_parameters["plot"]
            AcquisitionManager.row = f"row_{recording_parameters['row']}"
            row_path = os.path.join(recording_parameters["outputPath"], conf.customer_code, AcquisitionManager.plot, today,
                                    AcquisitionManager.row)
            AcquisitionManager.folder_index = tools.get_folder_index(row_path)

            AcquisitionManager.output_dir = os.path.join(row_path, str(AcquisitionManager.folder_index))

            if 'Default' in recording_parameters['configType']:
                acquisition_parameters = analysis_conf.acquisition_parameters
                output_types = acquisition_parameters.output_types
            else:
                acquisition_parameters = recording_parameters["Cameras"]
                output_types = acquisition_parameters["outputTypes"]

        output_types = [ot.lower() for ot in output_types]

        AcquisitionManager.output_clahe_fsi = 'clahe' in output_types or 'fsi' in output_types
        AcquisitionManager.output_equalize_hist_fsi = 'equalize_hist' in output_types or 'fsi' in output_types
        AcquisitionManager.output_rgb = 'rgb' in output_types
        AcquisitionManager.output_800 = '800' in output_types
        AcquisitionManager.output_975 = '975' in output_types
        AcquisitionManager.output_svo = 'svo' in output_types
        AcquisitionManager.output_zed_gray = 'zed_gray' in output_types
        AcquisitionManager.output_zed_depth = 'zed_depth' in output_types
        AcquisitionManager.output_zed_pc = 'zed_pc' in output_types
        AcquisitionManager.pass_clahe_stream = False
        AcquisitionManager.debug_mode = True

        alc_areas = [
            f"{v}{h}"
            for v in consts.alc_vertical_areas
            for h in consts.alc_horizontal_areas
        ]
        AcquisitionManager.alc_true_areas = acquisition_parameters.alc_true_areas
        AcquisitionManager.alc_false_areas = [a for a in alc_areas if a not in AcquisitionManager.alc_true_areas]

        AcquisitionManager.analyzer.set_output_dir(AcquisitionManager.output_dir)

        if not os.path.exists(AcquisitionManager.output_dir):
            os.makedirs(AcquisitionManager.output_dir)

    @staticmethod
    def receive_data():
        while not AcquisitionManager.shutdown_event.is_set():
            data, sender_module = AcquisitionManager.in_qu.get()
            action, data = data["action"], data["data"]
            if sender_module == ModulesEnum.GPS:
                if action == ModuleTransferAction.ENTER_PLOT and conf.autonomous_acquisition:
                    AcquisitionManager.plot = data
                    AcquisitionManager.start_recording(AcquisitionManager.plot, from_gps=True)
                    row_name = "/".join([
                        conf.customer_code, AcquisitionManager.plot,
                        AcquisitionManager.row, str(AcquisitionManager.folder_index)
                    ])
                    tools.log(f"START ACQUISITION FROM GPS - {row_name}/")
                    data = {
                        "plot": AcquisitionManager.plot,
                        "row": AcquisitionManager.get_row_number(AcquisitionManager.row),
                        "folder_index": AcquisitionManager.folder_index
                    }
                    AcquisitionManager.send_data(ModuleTransferAction.START_RECORDING, data, ModulesEnum.DataManager)
                elif action == ModuleTransferAction.EXIT_PLOT and conf.autonomous_acquisition:
                    tools.log("STOP ACQUISITION FROM GPS")
                    AcquisitionManager.stop_recording()
            elif sender_module == ModulesEnum.GUI:
                if action == ModuleTransferAction.START_RECORDING:
                    tools.log("START ACQUISITION FROM GUI")
                    AcquisitionManager.start_recording(recording_parameters=data, from_gps=False)
                    row_path = os.path.dirname(AcquisitionManager.output_dir)
                    data["folder_index"] = tools.get_folder_index(row_path, get_next_index=False)
                    AcquisitionManager.send_data(ModuleTransferAction.START_RECORDING, data, ModulesEnum.DataManager)
                elif action == ModuleTransferAction.STOP_RECORDING:
                    AcquisitionManager.stop_recording()
                    tools.log("STOP ACQUISITION FROM GUI")
            elif sender_module == ModulesEnum.Main:
                if action == ModuleTransferAction.MONITOR:
                    AcquisitionManager.send_data(
                        action=ModuleTransferAction.MONITOR,
                        data=None,
                        receiver=ModulesEnum.Main,
                        log_option=tools.LogOptions.NONE
                    )
                elif action == ModuleTransferAction.SET_LOGGER:
                    set_logger()

    @staticmethod
    def shutdown(sig, frame):
        tools.log(f"SHUTDOWN RECEIVED IN PROCESS {Module.module_name}", logging.WARNING)
        AcquisitionManager.shutdown_event.set()
        if not (AcquisitionManager.zed_connected and AcquisitionManager.jai_connected):
            AcquisitionManager.disconnect_cameras()
        if AcquisitionManager.running:
            AcquisitionManager.send_data(ModuleTransferAction.ACQUISITION_CRASH, None, ModulesEnum.GPS)
            AcquisitionManager.send_data(ModuleTransferAction.ACQUISITION_CRASH, None, ModulesEnum.DataManager)
        time.sleep(3)
