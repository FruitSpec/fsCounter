import os
import usb.core
import netifaces
import pypylon
import signal
import subprocess
from time import sleep
from GPS import location_awareness
from DataManager import uploader
from Analysis import analyzer
from utils.module_wrapper import ModuleManager, DataError, ModulesEnum
from utils.settings import conf
from GUI.gui_interface import GUIInterface
import jaized

global manager


def shutdown():
    manager[ModulesEnum.GPS].shutdown()
    manager[ModulesEnum.DataManager].shutdown()
    manager[ModulesEnum.Analysis].shutdown()


def transfer_data(sig, frame):
    for sender_module in ModulesEnum:
        try:
            data, recv_module = manager[sender_module].get_data()
            manager[recv_module].transfer_data(data, sender_module)
        except DataError:
            continue


def setup_GUI():
    def connect(sid, environ):
        gps_on = False
        try:
            jai_ip = netifaces.ifaddresses('eth2')[netifaces.AF_INET][0]['addr']
            jai_on = True
        except Exception:
            jai_on = False

        zed_on = False
        usbdev = usb.core.find(find_all=True)
        for dev in usbdev:
            try:
                manufacturer = usb.util.get_string(dev, dev.iManufacturer)
                product = usb.util.get_string(dev, dev.iProduct)
                if 'STEREOLABS' in manufacturer and 'ZED' in product:
                    dev.get_active_configuration()
                    zed_on = True
                    break
            except Exception:
                pass
        return jai_on, zed_on, gps_on

    def disconnect():
        print("disconnect GUI")

    def start_camera(mode, data=None):
        if mode == "record":
            if 'Default' in data['configType']:
                weather = data['weather']
                camera_data = conf["default camera data"][weather]
                output_types = conf["default output types"]
            else:
                camera_data = data["Cameras"]
                output_types = camera_data["outputTypes"]

            output_types = [ot.lower() for ot in output_types]

            fps = int(camera_data['FPS'])
            exposure_rgb = int(camera_data['IntegrationTimeRGB'])
            exposure_800 = int(camera_data['IntegrationTime800'])
            exposure_975 = int(camera_data['IntegrationTime975'])
            output_dir = os.path.join('/' + data['outputPath'], data['plot'], 'row_' + data['row'])
            output_fsi = 'fsi' in output_types
            output_rgb = 'rgb' in output_types
            output_800 = '800' in output_types
            output_975 = '975' in output_types
            output_svo = 'svo' in output_types
            view = False
            use_clahe_stretch = False
            debug_mode = True

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            jz_recorder = jaized.JaiZed()
            jz_recorder.connect_cameras(fps, exposure_rgb, exposure_800, exposure_975, output_dir,
                                        output_fsi, output_rgb, output_800, output_975, output_svo, view,
                                        use_clahe_stretch, debug_mode)
            jz_recorder.start_acquisition()
            return jz_recorder

    def stop_camera(jz_recorder):
        jz_recorder.stop_acquisition()
        jz_recorder.disconnect_cameras()

    GUIInterface.start_GUI(connect, disconnect, start_camera, stop_camera)


def main():
    global manager
    manager = dict()
    if conf["GUI"]:
        setup_GUI()
    for _, module in enumerate(ModulesEnum):
        manager[module] = ModuleManager()
    main_pid = os.getpid()
    manager[ModulesEnum.GPS].set_process(target=location_awareness.GPSSampler.init_module, main_pid=main_pid)
    manager[ModulesEnum.DataManager].set_process(target=uploader.init_module, main_pid=main_pid)
    manager[ModulesEnum.Analysis].set_process(target=analyzer.init_module, main_pid=main_pid)

    manager[ModulesEnum.GPS].start()
    manager[ModulesEnum.DataManager].start()
    manager[ModulesEnum.Analysis].start()

    signal.signal(signal.SIGUSR1, transfer_data)

    manager[ModulesEnum.GPS].join()
    manager[ModulesEnum.DataManager].join()
    manager[ModulesEnum.Analysis].join()


if __name__ == "__main__":
    main()
