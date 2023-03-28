import os
import usb.core
import netifaces
import pypylon
import signal
import subprocess
from time import sleep
import sys

sys.path.append("/home/mic-730ai/fruitspec/fsCounter/application")

from GPS.location_awareness import GPSSampler
from DataManager.data_manager import DataManager
from Analysis.acquisition_manager import AcquisitionManager
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
    global manager
    print("PID: ", os.getpid())
    print("handler: ", signal.getsignal(signal.SIGUSR1))
    for sender_module in ModulesEnum:
        try:
            data, recv_modules = manager[sender_module].retrieve_transferred_data()
            print(recv_modules)
            print(" -------------- ")
            for recv_module in recv_modules:
                print(recv_module)
                print(" $ $ $ $ $ $ $ $ $ $ ")
                manager[recv_module].receive_transferred_data(data, sender_module)
            break
        except DataError:
            continue


def main():
    global manager
    manager = dict()
    for _, module in enumerate(ModulesEnum):
        manager[module] = ModuleManager()
    main_pid = os.getpid()
    signal.signal(signal.SIGUSR1, transfer_data)

    print("MAIN PID:", main_pid)

    manager[ModulesEnum.GPS].set_process(
        target=GPSSampler.init_module,
        main_pid=main_pid,
        module_name=ModulesEnum.GPS
    )

    manager[ModulesEnum.GUI].set_process(
        target=GUIInterface.init_module,
        main_pid=main_pid,
        module_name=ModulesEnum.GUI
    )

    manager[ModulesEnum.DataManager].set_process(
        target=DataManager.init_module,
        main_pid=main_pid,
        module_name=ModulesEnum.DataManager
    )

    manager[ModulesEnum.Analysis].set_process(
        target=AcquisitionManager.init_module,
        main_pid=main_pid,
        module_name=ModulesEnum.Analysis
    )

    manager[ModulesEnum.GPS].start()
    manager[ModulesEnum.GUI].start()
    manager[ModulesEnum.DataManager].start()
    manager[ModulesEnum.Analysis].start()

    manager[ModulesEnum.GPS].join()
    manager[ModulesEnum.GUI].join()
    manager[ModulesEnum.DataManager].join()
    manager[ModulesEnum.Analysis].join()


if __name__ == "__main__":
    main()
