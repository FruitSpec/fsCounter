import os
from builtins import staticmethod
from application.utils.settings import GPS_conf, conf, analysis_conf, data_conf
from application.utils.module_wrapper import ModulesEnum, Module
import netifaces
import jaized


class AcquisitionManager(Module):
    jz_recorder = None
    fps = -1
    exposure_rgb, exposure_800, exposure_975 = -1, -1, -1
    output_dir = ""
    output_fsi, output_rgb, output_800, output_975, output_svo = False, False, False, False, False
    view, use_clahe_stretch, debug_mode = False, False, False

    @staticmethod
    def init_module(sender, receiver, main_pid):
        super(AcquisitionManager, AcquisitionManager).init_module(sender, receiver, main_pid)
        super(AcquisitionManager, AcquisitionManager).set_signals(AcquisitionManager.shutdown,
                                                                  AcquisitionManager.receive_data)
        AcquisitionManager.set_acquisition_parameters()
        AcquisitionManager.jz_recorder = jaized.JaiZed()
        AcquisitionManager.connect_cameras()

    @staticmethod
    def connect_cameras():
        AcquisitionManager.jz_recorder.connect_cameras(AcquisitionManager.fps, AcquisitionManager.exposure_rgb,
                                                       AcquisitionManager.exposure_800, AcquisitionManager.exposure_975,
                                                       AcquisitionManager.output_dir, AcquisitionManager.output_fsi,
                                                       AcquisitionManager.output_rgb, AcquisitionManager.output_800,
                                                       AcquisitionManager.output_975, AcquisitionManager.output_svo,
                                                       AcquisitionManager.view, AcquisitionManager.use_clahe_stretch,
                                                       AcquisitionManager.debug_mode)

    @staticmethod
    def set_acquisition_parameters(data=None):
        if not data:
            if analysis_conf["autonomous easy config"]:
                weather = analysis_conf["default weather"]
                camera_data = analysis_conf["default acquisition parameters"][weather]
            else:
                camera_data = analysis_conf["custom acquisition parameters"]
            output_types = camera_data["default output types"]
        else:
            if 'Default' in data['configType']:
                weather = data['weather']
                camera_data = conf["default camera data"][weather]
                output_types = conf["default output types"]
            else:
                camera_data = data["Cameras"]
                output_types = camera_data["outputTypes"]

        output_types = [ot.lower() for ot in output_types]

        AcquisitionManager.fps = int(camera_data['FPS'])
        AcquisitionManager.exposure_rgb = int(camera_data['IntegrationTimeRGB'])
        AcquisitionManager.exposure_800 = int(camera_data['IntegrationTime800'])
        AcquisitionManager.exposure_975 = int(camera_data['IntegrationTime975'])
        AcquisitionManager.output_dir = os.path.join(data_conf['output path'], 'jaized_app_temp')
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
        if sender_module == ModulesEnum.GPS:
            if data != GPS_conf["global polygon"]:
                AcquisitionManager.jz_recorder.start_acquisition()
            else:
                AcquisitionManager.jz_recorder.stop_acquisition()
        elif sender_module == ModulesEnum.GUI:
            if data['action'] == 'start':
                data = data['data']
                AcquisitionManager.set_acquisition_parameters(data)

