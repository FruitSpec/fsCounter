import os
import signal
import subprocess
from time import sleep
from GPS import location_awareness
from DataManager import uploader
from Analysis import analyzer
from utils.module_wrapper import ModuleManager, DataError, ModulesEnum
from utils.settings import conf
from GUI.gui_interface import GUIInterface

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
        pass

    def disconnect(sid, environ):
        pass

    def start_camera(sid, environ):
        pass

    def stop_camera(sid, environ):
        pass

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
