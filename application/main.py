import os
import signal
import logging
from threading import Lock
from multiprocessing import Queue
import time
import sys

sys.path.append("/home/mic-730ai/fruitspec/fsCounter/application")

from application.utils.settings import set_logger

set_logger()

from GPS.location_awareness import GPSSampler
from DataManager.data_manager import DataManager
from Analysis.acquisition_manager import AcquisitionManager
from Analysis.alternative_flow import AlternativeFlow
from utils.module_wrapper import ModuleManager, DataError, ModulesEnum
from GUI.gui_interface import GUIInterface

global manager, communication_queue, transfer_data_lock


def shutdown():
    manager[ModulesEnum.GPS].shutdown()
    manager[ModulesEnum.DataManager].shutdown()
    manager[ModulesEnum.Analysis].shutdown()


def transfer_data(sig, frame):
    global manager, communication_queue, transfer_data_lock
    with transfer_data_lock:
        sender_module = communication_queue.get()
        for i in range(5):
            try:
                data, recv_module = manager[sender_module].retrieve_transferred_data()
                manager[recv_module].receive_transferred_data(data, sender_module)
                break
            except DataError:
                time.sleep(0.1)
                logging.exception("communication error ", i)


def main():
    global manager, communication_queue, transfer_data_lock
    manager = dict()
    communication_queue = Queue()
    transfer_data_lock = Lock()
    for _, module in enumerate(ModulesEnum):
        manager[module] = ModuleManager()
    main_pid = os.getpid()
    signal.signal(signal.SIGUSR1, transfer_data)

    logging.info(f"MAIN PID: {main_pid}")
    print(f"MAIN PID: {main_pid}")

    manager[ModulesEnum.GPS].set_process(
        target=GPSSampler.init_module,
        main_pid=main_pid,
        module_name=ModulesEnum.GPS,
        communication_queue=communication_queue
    )

    manager[ModulesEnum.GUI].set_process(
        target=GUIInterface.init_module,
        main_pid=main_pid,
        module_name=ModulesEnum.GUI,
        communication_queue=communication_queue
    )

    manager[ModulesEnum.DataManager].set_process(
        target=DataManager.init_module,
        main_pid=main_pid,
        module_name=ModulesEnum.DataManager,
        communication_queue=communication_queue
    )

    manager[ModulesEnum.Acquisition].set_process(
        target=AcquisitionManager.init_module,
        main_pid=main_pid,
        module_name=ModulesEnum.Acquisition,
        communication_queue=communication_queue,
        daemon=False
    )

    manager[ModulesEnum.Analysis].set_process(
        target=AlternativeFlow.init_module,
        main_pid=main_pid,
        module_name=ModulesEnum.Analysis,
        communication_queue=communication_queue
    )

    manager[ModulesEnum.GPS].start()
    manager[ModulesEnum.GUI].start()
    manager[ModulesEnum.DataManager].start()
    manager[ModulesEnum.Acquisition].start()
    manager[ModulesEnum.Analysis].start()

    manager[ModulesEnum.GPS].join()
    manager[ModulesEnum.GUI].join()
    manager[ModulesEnum.DataManager].join()
    manager[ModulesEnum.Acquisition].join()
    manager[ModulesEnum.Analysis].join()


if __name__ == "__main__":
    main()
