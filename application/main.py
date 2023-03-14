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
        jz = jaized.JaiZed()
        jz.connect_cameras_wra()
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
        if jai_on and zed_on:

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
            output_path = os.path.join('/' + data['outputPath'], data['plot'], 'row_' + data['row'])

            recording_params = f" --FPS {camera_data['FPS']} "
            recording_params += f"--output-dir {output_path} "
            recording_params += f"--exposure-rgb {camera_data['IntegrationTimeRGB']} "
            recording_params += f"--exposure-800 {camera_data['IntegrationTime800']} "
            recording_params += f"--exposure-975 {camera_data['IntegrationTime975']} "

            if not os.path.exists(output_path):
                os.makedirs(output_path)
            for ot in output_types:
                recording_params += f'--output-{ot.lower()} '
            return proc.pid

    def stop_camera(pid):
        if pid > 0:
            os.killpg(os.getpgid(pid), signal.SIGTERM)

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
