import glob
import os
import pandas as pd
import logging
import subprocess
import threading
import traceback
from multiprocessing import Queue
from application.utils.settings import conf, data_conf, consts
from application.utils import tools
from datetime import datetime, timedelta
import psutil
import shutil
import time
import sys

sys.path.append("/home/mic-730ai/fruitspec/fsCounter/application")

from application.utils.settings import set_logger

global logger_date
logger_date = set_logger()

from GPS.location_awareness import GPSSampler
from DataManager.data_manager import DataManager
from Analysis.acquisition_manager import AcquisitionManager
from Analysis.alternative_flow import AlternativeFlow
from utils.module_wrapper import ModuleManager, DataError, ModulesEnum, ModuleTransferAction
from GUI.gui_interface import GUIInterface

global manager, communication_queue, process_monitor_events


def storage_cleanup():

    def get_total_path(x, name_only=False):
        name = os.path.join(
            str(x["customer_code"]), str(x["plot_code"]),
            str(x["scan_date"]), f"row_{x['row']}", str(x["folder_index"])
        )
        if name_only:
            return name
        else:
            return os.path.join(data_conf.output_path, name)

    def get_creation_date(path):
            try:
                return os.path.getctime(path)
            except:
                return None

    def routine_cleanup():
        tools.log("PERFORMING ROUTINE CLEANUP...")

        try:
            uploaded_df = pd.read_csv(data_conf.uploaded_path, dtype=str).dropna(how="any")
        except (FileNotFoundError, IOError):
            tools.log("ROUTINE CLEANUP DONE")
            return pd.DataFrame(dtype=str)

        if uploaded_df.empty:
            tools.log("ROUTINE CLEANUP DONE")
            return pd.DataFrame(dtype=str)

        uploaded_df["total_path"] = uploaded_df.apply(get_total_path, axis=1)
        uploaded_df["creation_date"] = uploaded_df["total_path"].apply(get_creation_date)

        # Filter the DataFrame to delete files older than 48 hours
        cutoff_time = datetime.now() - timedelta(hours=data_conf.routine_delete_time_hours)
        is_old = uploaded_df['creation_date'] <= cutoff_time.timestamp()
        not_exist = pd.isna(uploaded_df['creation_date'])
        uploaded_delete = uploaded_df[is_old | not_exist]

        for _, r in uploaded_delete.iterrows():
            if pd.isna(r["creation_date"]):
                continue
            total_path = r["total_path"]
            row_name = get_total_path(r, name_only=True)
            row_folder = os.path.dirname(total_path)
            try:
                shutil.rmtree(row_folder)
                tools.log(f"ROUTINE STORAGE CLEANUP - DELETED {row_name}")
            except (FileNotFoundError, IOError):
                tools.log(f"ROUTINE STORAGE CLEANUP - COULD NOT DELETE {row_name}", logging.WARNING)

        uploaded_delete.drop(columns=["total_path", "creation_date"], inplace=True)
        uploaded_delete["procedure_type"] = consts.routine

        tools.log("ROUTINE CLEANUP DONE")

        return uploaded_delete

    def urgent_cleanup():

        def get_folder_size(path):
            try:
                return sum([os.path.getsize(f) for f in os.scandir(path)]) / (1024 ** 2)
            except:
                return -1

        disk_occupancy = psutil.disk_usage("/").percent

        tools.log("PERFORMING URGENT CLEANUP...")
        tools.log(f"DISK OCCUPANCY {disk_occupancy}%")
        tools.log(f"DISK OCCUPANCY THRESHOLD {data_conf.max_disk_occupancy}%")

        analyzed_delete = None
        if disk_occupancy > data_conf.max_disk_occupancy:
            try:
                analyzed_df = pd.read_csv(data_conf.analyzed_path, dtype=str).dropna(how="any")
            except (FileNotFoundError, IOError):
                tools.log("DISK TOO FULL BUT NO FILES TO DELETE!", logging.WARNING)
                tools.log("URGENT CLEANUP DONE")
                return pd.DataFrame()

            if analyzed_df.empty:
                tools.log("DISK TOO FULL BUT NO FILES TO DELETE!", logging.WARNING)
                tools.log("URGENT CLEANUP DONE")
                return pd.DataFrame()

            analyzed_df["total_path"] = analyzed_df.apply(get_total_path, axis=1)

            analyzed_df["file_size_in_MB"] = analyzed_df["total_path"].apply(get_folder_size)
            invalid_df = analyzed_df[analyzed_df["file_size_in_MB"] == -1]
            analyzed_df = analyzed_df[analyzed_df["file_size_in_MB"] >= data_conf.storage_cleanup_size_threshold]
            analyzed_df.sort_values(by="file_size_in_MB", inplace=True)
            analyzed_groups = analyzed_df.groupby(["customer_code", "plot_code", "scan_date"])

            analyzed_delete = pd.DataFrame(columns=analyzed_df.columns)

            for _, analyzed_gr in analyzed_groups:
                if len(analyzed_gr) >= data_conf.storage_cleanup_min_files:
                    delete_from_gr = analyzed_gr.iloc[data_conf.storage_cleanup_min_files:]
                    analyzed_delete = pd.concat([analyzed_delete, delete_from_gr])

            if analyzed_delete.empty:
                tools.log("DISK TOO FULL BUT NO FILES TO DELETE!", logging.WARNING)
            else:
                for _, r in analyzed_delete.iterrows():
                    total_path = r["total_path"]
                    row_name = get_total_path(r, name_only=True)
                    try:
                        files_to_delete = glob.glob(os.path.join(total_path, "*.svo")) \
                                          + glob.glob(os.path.join(total_path, "*.mkv"))
                        for f in files_to_delete:
                            os.remove(f)
                        tools.log(f"URGENT STORAGE CLEANUP - DELETED {row_name}")

                    except (FileNotFoundError, IOError):
                        tools.log(f"URGENT STORAGE CLEANUP - COULD NOT DELETE {row_name}", logging.WARNING)
                    if psutil.disk_usage("/").percent < data_conf.min_disk_occupancy:
                        break

            analyzed_delete.drop(columns=["total_path", "file_size_in_MB"], inplace=True)
            analyzed_delete["procedure_type"] = consts.urgent
            invalid_df["procedure_type"] = "FileNotFound"

            analyzed_delete = pd.concat([analyzed_delete, invalid_df])

        tools.log("URGENT CLEANUP DONE")
        return analyzed_delete

    def rewrite_not_deleted(path):
        nonlocal deleted_df
        try:
            df = pd.read_csv(path, dtype=str)
        except (FileNotFoundError, IOError):
            return

        not_deleted_df = pd.merge(df, deleted_df.astype(str), how='left', indicator=True,
                                  on=["customer_code", "plot_code", "scan_date", "row", "folder_index"])

        not_deleted_df = not_deleted_df[not_deleted_df['_merge'] == 'left_only']
        not_deleted_df = not_deleted_df[df.columns]

        not_deleted_df.to_csv(path, header=True, index=False)

    r_delete = routine_cleanup()
    u_delete = urgent_cleanup()

    deleted_df = pd.concat([r_delete, u_delete])
    if deleted_df.empty:
        return
    deleted_df = deleted_df[["customer_code", "plot_code", "scan_date", "row", "folder_index", "procedure_type"]]

    is_first = not os.path.exists(data_conf.deleted_path)
    deleted_df.to_csv(data_conf.deleted_path, mode='a+', header=is_first, index=False)
    deleted_df = pd.read_csv(data_conf.deleted_path, dtype=str)
    deleted_df = deleted_df[deleted_df["procedure_type"] == consts.routine]

    rewrite_not_deleted(data_conf.collected_path)
    rewrite_not_deleted(data_conf.analyzed_path)
    rewrite_not_deleted(data_conf.uploaded_path)


def restart_application(startup_count, startup_time, reboot=False):
    time.sleep(3)
    global manager

    time_since_startup = time.time() - startup_time

    if time_since_startup < consts.restart_time_threshold:
        startup_count += 1
    else:
        startup_count = 1

    for k in manager:
        try:
            manager[k].terminate()
        except:
            pass

    time.sleep(3)

    for k in manager:
        try:
            manager[k].force_kill()
        except:
            pass

    time.sleep(1)

    if (not reboot) and startup_count <= consts.restart_count_threshold:
        tools.log(f"APPLICATION RESTARTING - NEW STARTUP COUNT: {startup_count}")
        os.execl("/bin/bash", "/bin/bash", consts.startup_script, str(startup_count))
    else:
        tools.log(f"SYSTEM REBOOTING - STARTUP COUNT: {startup_count}")
        os.system("reboot")


def process_monitor(startup_count, startup_time):
    global manager, logger_date
    time.sleep(40)
    while True:
        # today = datetime.now().strftime('%d%m%y')
        # if today != logger_date:
        #     set_logger()
        #     for k in manager:
        #         send_data_to_module(ModuleTransferAction.SET_LOGGER, None, k)

        tools.log("MONITORING MODULES...", log_option=tools.LogOptions.LOG)
        for k in manager:
            if (not conf.GUI and k == ModulesEnum.GUI) or k == ModulesEnum.Main:
                continue
            process_monitor_events[k].clear()
            send_data_to_module(ModuleTransferAction.MONITOR, None, k, log_option=tools.LogOptions.LOG)

        for k in manager:
            if (not conf.GUI and k == ModulesEnum.GUI) or k == ModulesEnum.Main:
                continue
            if not manager[k].is_alive():
                alive = False
                death_cause = consts.process_not_alive
            else:
                process_monitor_events[k].wait(2)
                alive = process_monitor_events[k].is_set()
                death_cause = consts.process_not_responding
            if not alive:
                tools.log(f"PROCESS {k} IS DEAD - {death_cause} - RESPAWNING...", logging.WARNING)
                try:
                    for receiver in manager[k].notify_on_death:
                        send_data_to_module(
                            action=manager[k].death_action,
                            data=None,
                            receiver=receiver,
                            sender=k,
                            log_option=tools.LogOptions.BOTH
                        )
                except TypeError:
                    pass
                except Exception as e:
                    tools.log(f"COULD NOT NOTIFY ON {k}'s DEATH - {e}", logging.WARNING)

                # manager[k].respawn()
                tools.log(f"RESTARTING APPLICATION", logging.info)
                restart_application(startup_count, startup_time)
                return

        tools.log("MONITORING MODULES - OK", log_option=tools.LogOptions.LOG)
        time.sleep(5)


def send_data_to_module(action, data, receiver, sender=ModulesEnum.Main, log_option=tools.LogOptions.NONE):
    data = {
        "action": action,
        "data": data
    }
    tools.log(
        f"DATA TRANSFER:\n\t"
        f"FROM {ModulesEnum.Main} AS {sender}\n\t"
        f"TO {receiver}\n\t"
        f"ACTION: {action}",
        log_option=log_option
    )
    manager[receiver].receive_transferred_data(data, sender)


def transfer_data(startup_count, startup_time):
    global manager, communication_queue, process_monitor_events

    while True:
        sender_module = communication_queue.get()
        success, retrieved = False, False
        receiver, action, err_msg = None, None, None

        try:
            data, receiver = manager[sender_module].retrieve_transferred_data()
            retrieved = True
            log_option = data["log_option"]
            action = data["action"]
            tools.log(
                f"DATA TRANSFER:\n\t"
                f"FROM {sender_module}\n\t"
                f"TO {receiver}\n\t"
                f"ACTION: {action}",
                log_option=log_option
            )
            if receiver == ModulesEnum.Main:
                if action == ModuleTransferAction.MONITOR:
                    process_monitor_events[sender_module].set()
                elif action == ModuleTransferAction.RESTART_APP:
                    restart_application(startup_count, startup_time)
                elif action == ModuleTransferAction.REBOOT:
                    restart_application(startup_count, startup_time, reboot=True)
            else:
                manager[receiver].receive_transferred_data(data, sender_module)
            success = True
        except DataError:
            err_msg = "DATA ERROR"
        except ProcessLookupError:
            err_msg = "PROCESS LOOKUP ERROR"
        except Exception as e:
            err_msg = "UNKNOWN ERROR: " + str(e)
            tools.log("!UNKNOWN ERROR!", logging.ERROR, exc_info=True)
        finally:
            if not success:
                if not retrieved:
                    communication_queue.put(sender_module)
                else:
                    tools.log(f"IPC WARNING - SIGNAL LOST", logging.WARNING)

                tools.log(
                    f"IPC FAILURE\n\t"
                    f"FROM {sender_module}\n\t"
                    f"TO {receiver}\n\t"
                    f"ACTION {action}\n\t"
                    f"ERROR {err_msg}",
                    log_level=logging.WARNING
                )


def main():
    global manager, communication_queue, process_monitor_events

    try:
        startup_count = int(sys.argv[1])
    except:
        startup_count = 1

    startup_time = time.time()

    manager = dict()
    process_monitor_events = dict()
    communication_queue = Queue()
    main_pid = os.getpid()

    tools.log(f"APPLICATION STARTING - STARTUP COUNT: {startup_count}")
    tools.log(f"MAIN PID: {main_pid}")

    for _, module in enumerate(ModulesEnum):
        if module != ModulesEnum.Main:
            manager[module] = ModuleManager(main_pid, communication_queue)
            process_monitor_events[module] = threading.Event()

    storage_cleanup()

    transfer_data_t = threading.Thread(target=transfer_data, args=(startup_count, startup_time))
    transfer_data_t.start()

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
        death_action=ModuleTransferAction.ACQUISITION_CRASH
    )

    manager[ModulesEnum.Analysis].set_process(
        target=AlternativeFlow.init_module,
        module_name=ModulesEnum.Analysis,
    )

    acquisition_pid = manager[ModulesEnum.Acquisition].start()
    analysis_pid = manager[ModulesEnum.Analysis].start()
    data_manager_pid = manager[ModulesEnum.DataManager].start()
    gui_pid = manager[ModulesEnum.GUI].start()
    gps_pid = manager[ModulesEnum.GPS].start()

    process_monitor_t = threading.Thread(target=process_monitor, args=(startup_count, startup_time), daemon=True)
    process_monitor_t.start()

    manager[ModulesEnum.GPS].join()
    manager[ModulesEnum.GUI].join()
    manager[ModulesEnum.DataManager].join()
    manager[ModulesEnum.Acquisition].join()
    manager[ModulesEnum.Analysis].join()


if __name__ == "__main__":
    main()
