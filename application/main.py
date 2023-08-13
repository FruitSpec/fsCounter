import glob
import os
import pandas as pd
import logging
import subprocess
import threading
import traceback
from multiprocessing import Queue
from application.utils.settings import conf, data_conf, consts
from datetime import datetime, timedelta
import psutil
import shutil
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

global manager, communication_queue, monitor_events


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
    time.sleep(5)
    logging.info("REBOOT")
    print("REBOOT")
    os.system("reboot")


def process_monitor():
    global manager
    time.sleep(60)
    while True:
        logging.info("MONITORING MODULES")
        for k in manager:
            if (not conf.GUI and k == ModulesEnum.GUI) or k == ModulesEnum.Main:
                continue
            monitor_events[k].clear()
            send_data_to_module(ModuleTransferAction.MONITOR, None, k)

        for k in manager:
            if (not conf.GUI and k == ModulesEnum.GUI) or k == ModulesEnum.Main:
                continue
            if manager[k].is_alive():
                monitor_events[k].wait(2)
                alive = monitor_events[k].is_set()
            else:
                alive = False
            if not alive:
                logging.warning(f"PROCESS {k} IS DEAD - RESPAWNING...")
                try:
                    for recv_module in manager[k].notify_on_death:
                        send_data_to_module(manager[k].death_action, None, recv_module)
                except TypeError:
                    pass
                # manager[k].respawn()
                restart_application(killer=k)
                return
        time.sleep(5)

def send_data_to_module(action, data, recv_module):
    data = {
        "action": action,
        "data": data
    }
    manager[recv_module].receive_transferred_data(data, ModulesEnum.Main)


def transfer_data():
    global manager, communication_queue, monitor_events

    while True:
        sender_module = communication_queue.get()
        success, retrieved = False, False
        recv_module, action, err_msg = None, None, None

        try:
            data, recv_module = manager[sender_module].retrieve_transferred_data()
            retrieved = True
            action = data["action"]
            logging.info(
                f"DATA TRANSFER:\n\t"
                f"FROM {sender_module}\n\t"
                f"TO {recv_module}\n\t"
                f"ACTION: {action}"
            )
            if recv_module == ModulesEnum.Main:
                if action == ModuleTransferAction.MONITOR:
                    monitor_events[sender_module].set()
                elif action == ModuleTransferAction.RESTART_APP:
                    restart_application(sender_module)
            else:
                manager[recv_module].receive_transferred_data(data, sender_module)
            success = True
        except DataError:
            err_msg = "DATA ERROR"
        except ProcessLookupError:
            err_msg = "PROCESS LOOKUP ERROR"
        except Exception as e:
            err_msg = "UNKNOWN ERROR: " + str(e)
            logging.exception("!UNKNOWN ERROR!")
            traceback.print_exc()
        finally:
            if not success:
                if not retrieved:
                    communication_queue.put(sender_module)
                else:
                    logging.warning(f"IPC WARNING - SIGNAL LOST")

                logging.warning(
                    f"IPC FAILURE\n\t"
                    f"FROM {sender_module}\n\t"
                    f"TO {recv_module}\n\t"
                    f"ACTION {action}\n\t"
                    f"ERROR {err_msg}"
                )
                print(
                    f"IPC FAILURE\n\t"
                    f"FROM {sender_module}\n\t"
                    f"TO {recv_module}\n\t"
                    f"ACTION {action}\n\t"
                    f"ERROR {err_msg}"
                )


def main():
    global manager, communication_queue, monitor_events
    manager = dict()
    monitor_events = dict()
    communication_queue = Queue()
    main_pid = os.getpid()

    for _, module in enumerate(ModulesEnum):
        if module != ModulesEnum.Main:
            manager[module] = ModuleManager(main_pid, communication_queue)
            monitor_events[module] = threading.Event()

    transfer_data_t = threading.Thread(target=transfer_data)
    transfer_data_t.start()

    print(f"MAIN PID: {main_pid}")
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
