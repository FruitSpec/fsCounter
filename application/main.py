import os
import queue
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
from utils.module_wrapper import ModuleManager, DataError, ModulesEnum, ModuleTransferAction
from GUI.gui_interface import GUIInterface

global manager, communication_queue, transfer_data_lock


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


def restart_application(killer=None):
    time.sleep(2)
    global manager
    for k in manager:
        if not conf.GUI and k == ModulesEnum.GUI:
            continue
        elif not killer or k != killer:
            manager[k].terminate()
    time.sleep(2)
    logging.info("REBOOT")
    print("REBOOT")
    os.system("reboot")


def process_monitor():
    global manager

    while True:
        logging.info("MONITORING MODULES")
        for k in manager:
            if not conf.GUI and k == ModulesEnum.GUI:
                continue
            elif not manager[k].is_alive():
                logging.warning(f"PROCESS {k} IS DEAD - RESPAWNING...")
                try:
                    for recv_module in manager[k].notify_on_death:
                        data = {
                            "action": manager[k].death_action,
                            "data": None
                        }
                        manager[recv_module].receive_transferred_data(data, k)
                except TypeError:
                    pass
                # manager[k].respawn()
                restart_application(killer=k)
                return
        time.sleep(5)


def transfer_data(sig, frame):
    global manager, communication_queue, transfer_data_lock

    is_acquired = transfer_data_lock.acquire(timeout=1)
    if not is_acquired:
        logging.warning("TRANSFER DATA LOCK COULD NOT BE ACQUIRED")
        print("TRANSFER DATA LOCK COULD NOT BE ACQUIRED")
        restart_application()
        return
    try:
        sender_module = communication_queue.get(timeout=1)
    except queue.Empty:
        logging.warning("COMMUNICATION ERROR - COMMUNICATION QUEUE IS EMPTY")
        print("COMMUNICATION ERROR - COMMUNICATION QUEUE IS EMPTY")
        restart_application()
        return

    success = False
    recv_module = None
    action = None
    for i in range(5):
        try:
            data, recv_module = manager[sender_module].retrieve_transferred_data()
            action = data["action"]
            logging.info(f"DATA TRANSFER:\n\tFROM {sender_module}\n\tTO {recv_module}\n\tACTION: {action}")
            manager[recv_module].receive_transferred_data(data, sender_module)
            success = True
            break
        except DataError:
            time.sleep(0.1)
            logging.warning(f"COMMUNICATION ERROR #{i}")
            print(f"COMMUNICATION ERROR #{i}")
        except ProcessLookupError:
            logging.exception(f"PROCESS LOOKUP ERROR: ")
            print(f"PROCESS LOOKUP ERROR: ")
            traceback.print_exc()
            break
        except Exception:
            logging.exception(f"UNKNOWN COMMUNICATION ERROR: ")
            print(f"UNKNOWN COMMUNICATION ERROR: ")
            traceback.print_exc()

    transfer_data_lock.release()
    if not success:
        logging.warning(f"IPC FAILURE - FROM {sender_module} TO {recv_module} WITH ACTION {action}")
        print(f"IPC FAILURE - FROM {sender_module} TO {recv_module} WITH ACTION {action}")


def main():
    global manager, communication_queue
    manager = dict()
    communication_queue = Queue()
    main_pid = os.getpid()
    for _, module in enumerate(ModulesEnum):
        manager[module] = ModuleManager(main_pid, communication_queue)
    print(f"MAIN PID: {main_pid}")

    # transfer_data_t = threading.Thread(target=transfer_data)
    signal.signal(signal.SIGUSR1, transfer_data)

    logging.info(f"MAIN PID: {main_pid}")

    manager[ModulesEnum.GPS].set_process(
        target=GPSSampler.init_module,
        module_name=ModulesEnum.GPS
    )

    manager[ModulesEnum.GUI].set_process(
        target=GUIInterface.init_module,
        module_name=ModulesEnum.GUI
    )

    manager[ModulesEnum.DataManager].set_process(
        target=DataManager.init_module,
        module_name=ModulesEnum.DataManager
    )

    manager[ModulesEnum.Acquisition].set_process(
        target=AcquisitionManager.init_module,
        module_name=ModulesEnum.Acquisition,
        notify_on_death=[ModulesEnum.GPS, ModulesEnum.DataManager],
        death_action=ModuleTransferAction.ACQUISITION_CRASH,
        daemon=False
    )

    manager[ModulesEnum.Analysis].set_process(
        target=AlternativeFlow.init_module,
        module_name=ModulesEnum.Analysis,
    )

    gps_pid = manager[ModulesEnum.GPS].start()
    gui_pid = manager[ModulesEnum.GUI].start()
    data_manager_pid = manager[ModulesEnum.DataManager].start()
    acquisition_pid = manager[ModulesEnum.Acquisition].start()
    analysis_pid = manager[ModulesEnum.Analysis].start()

    monitor_t = threading.Thread(target=process_monitor, daemon=True)
    monitor_t.start()

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
