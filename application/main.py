import os
import signal
import logging
import subprocess
import threading
import traceback
from threading import Lock
from multiprocessing import Queue
from application.utils.settings import conf
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


def start_strace(main_pid, gps_pid, gui_pid, data_manager_pid, acquisition_pid, analysis_pid):

    def write_pid_to_file(pid, file_path, cmd):
        with open(file_path, mode="a+") as f:
            f.write(f"\nSTARTED WITH PID {pid}\n\n")
            subprocess.Popen(cmd, stdout=f, stderr=f)

    strace_output_path = "/home/mic-730ai/Desktop/strace/"
    main_output = strace_output_path + "main_strace.txt"
    gps_output = strace_output_path + "gps_strace.txt"
    gui_output = strace_output_path + "gui_strace.txt"
    data_manager_output = strace_output_path + "data_manager_strace.txt"
    acquisition_output = strace_output_path + "acquisition_strace.txt"
    analysis_output = strace_output_path + "analysis_strace.txt"

    command = ['strace', '-t', '-e', 'trace=signal', '-p']
    main_cmd = command + [str(main_pid)]
    gps_cmd = command + [str(gps_pid)]
    gui_cmd = command + [str(gui_pid)]
    data_manager_cmd = command + [str(data_manager_pid)]
    acquisition_cmd = command + [str(acquisition_pid)]
    analysis_cmd = command + [str(analysis_pid)]

    write_pid_to_file(main_pid, main_output, main_cmd)
    write_pid_to_file(gps_pid, gps_output, gps_cmd)
    write_pid_to_file(gui_pid, gui_output, gui_cmd)
    write_pid_to_file(data_manager_pid, data_manager_output, data_manager_cmd)
    write_pid_to_file(acquisition_pid, acquisition_output, acquisition_cmd)
    write_pid_to_file(analysis_pid, analysis_output, analysis_cmd)


def transfer_data(sig, frame):
    global manager, communication_queue, transfer_data_lock
    with transfer_data_lock:
        sender_module = communication_queue.get()
        logging.info(f"DATA TRANSFER FROM {sender_module}")
        success = False
        for i in range(5):
            try:
                data, recv_module = manager[sender_module].retrieve_transferred_data()
                logging.info(f"DATA TRANSFER TO {recv_module}")
                manager[recv_module].receive_transferred_data(data, sender_module)
                success = True
                break
            except DataError:
                time.sleep(0.1)
                logging.warning(f"COMMUNICATION ERROR #{i}")
            except ProcessLookupError:
                success = recv_module == ModulesEnum.GUI
                if not success:
                    logging.exception(f"PROCESS LOOKUP ERROR: ")
                    traceback.print_exc()
            except:
                logging.exception(f"UNKNOWN COMMUNICATION ERROR: ")
                traceback.print_exc()
        if not success:
            logging.warning(f"IPC FAILURE - FROM {sender_module} TO {recv_module} WITH ACTION {data['action']}")
            print(f"IPC FAILURE - FROM {sender_module} TO {recv_module} WITH ACTION {data['action']}")


def main():
    global manager, communication_queue, transfer_data_lock
    manager = dict()
    communication_queue = Queue()
    transfer_data_lock = Lock()
    for _, module in enumerate(ModulesEnum):
        manager[module] = ModuleManager()
    main_pid = os.getpid()
    print(f"MAIN PID: {main_pid}")
    signal.signal(signal.SIGUSR1, transfer_data)

    logging.info(f"MAIN PID: {main_pid}")

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

    gps_pid = manager[ModulesEnum.GPS].start()
    gui_pid = manager[ModulesEnum.GUI].start()
    data_manager_pid = manager[ModulesEnum.DataManager].start()
    acquisition_pid = manager[ModulesEnum.Acquisition].start()
    analysis_pid = manager[ModulesEnum.Analysis].start()

    if conf.use_strace:
        strace_t = threading.Thread(target=start_strace, daemon=True,
                                    args=(main_pid, gps_pid, gui_pid, data_manager_pid, acquisition_pid, analysis_pid))
        strace_t.start()

    manager[ModulesEnum.GPS].join()
    manager[ModulesEnum.GUI].join()
    manager[ModulesEnum.DataManager].join()
    manager[ModulesEnum.Acquisition].join()
    manager[ModulesEnum.Analysis].join()


if __name__ == "__main__":
    main()
