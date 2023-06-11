import os.path
import threading
from builtins import staticmethod
import time
from datetime import datetime
import signal
import logging
import boto3
import pandas as pd
from requests.exceptions import RequestException
from application.utils.settings import data_conf, conf
from application.utils.module_wrapper import ModulesEnum, Module, ModuleTransferAction
import application.utils.tools as tools
import speedtest


class DataManager(Module):
    previous_plot, current_plot = data_conf["global polygon"], data_conf["global polygon"]
    current_row = -1
    current_path, current_index = None, -1
    fruits_data = dict()
    fruits_data_lock, scan_lock = threading.Lock(), threading.Lock()
    s3_client = None
    update_output_thread, internet_scan_thread = None, None
    scan_df = pd.DataFrame(data={"customer_code": [], "plot_code": [], "scan_date": [], "row": [], "filename": []})
    collected_df = pd.DataFrame(
        data={"customer_code": [], "plot_code": [], "scan_date": [], "row": [], "file_index": []})
    analyzed_df = pd.DataFrame(
        data={"customer_code": [], "plot_code": [], "scan_date": [], "row": [], "file_index": []})

    @staticmethod
    def init_module(sender, receiver, main_pid, module_name):
        super(DataManager, DataManager).init_module(sender, receiver, main_pid, module_name)
        super(DataManager, DataManager).set_signals(DataManager.shutdown, DataManager.receive_data)

        DataManager.s3_client = boto3.client("s3")
        DataManager.update_output_thread = threading.Thread(target=DataManager.update_output, daemon=True)
        DataManager.internet_scan_thread = threading.Thread(target=DataManager.internet_scan, daemon=True)

        try:
            DataManager.collected_df = pd.read_csv(data_conf["collected path"], dtype=str)
        except FileNotFoundError:
            DataManager.collected_df = pd.DataFrame(
                data={"customer_code": [], "plot_code": [], "scan_date": [], "row": [], "file_index": []})

        try:
            DataManager.analyzed_df = pd.read_csv(data_conf["analyzed path"], dtype=str)
        except FileNotFoundError:
            DataManager.analyzed_df = pd.DataFrame(
                data={"customer_code": [], "plot_code": [], "scan_date": [], "row": [], "file_index": []})

        DataManager.update_output_thread.start()
        DataManager.internet_scan_thread.start()
        DataManager.update_output_thread.join()
        DataManager.internet_scan_thread.join()

    @staticmethod
    def start_new_file():
        def add_to_scan():
            """ add the recently closed file to the scan df to prepare for uploading """
            filename_csv = f"fruits_{DataManager.current_index}.csv"
            if not os.path.exists(DataManager.current_path):
                logging.info(f"CSV PATH : {DataManager.current_path} - NOT EXIST - NO DATA DETECTED")

            if data_conf["use feather"]:
                # convert CSV to feather before trying to upload to S3. should take 2-3 seconds if file size < 1MB.
                filename_feather = f"fruits_{DataManager.current_index}.feather"
                feather_path = tools.get_fruits_path(
                    plot=DataManager.current_plot, row=DataManager.current_row,
                    index=DataManager.current_index, write_csv=False
                )
                pd.read_csv(DataManager.current_path).to_feather(feather_path)
                filename = filename_feather
            else:
                filename = filename_csv
            today = datetime.now().strftime("%d%m%y")
            data = {
                "customer code": [conf["customer code"]],
                "plot code": [DataManager.current_plot],
                "scan date": [today],
                "row": [DataManager.current_row],
                "filename": [filename]
            }
            tmp_df = pd.DataFrame(data=data, index=[0])
            DataManager.scan_lock.acquire(blocking=True)
            DataManager.scan_df = pd.concat([DataManager.scan_df, tmp_df], axis=0)
            DataManager.scan_df.drop_duplicates(inplace=True)
            logging.info(f"PREVIOUS FILE {DataManager.current_plot}/{filename} ADDED TO SCAN DF")
            DataManager.scan_lock.release()

        if DataManager.current_path:
            add_to_scan()

    @staticmethod
    def receive_data(sig, frame):
        data, sender_module = DataManager.receiver.recv()
        action, data = data["action"], data["data"]
        if sender_module == ModulesEnum.GPS:
            if action == ModuleTransferAction.BLOCK_SWITCH and data != DataManager.current_plot:
                # entering a new block which is not the latest block we've been to
                logging.info(f"NEW BLOCK ENTRANCE - {data}")
                DataManager.write_fruits_data_locally(release=False)
                DataManager.previous_plot = DataManager.current_plot
                DataManager.current_plot = data
                DataManager.start_new_file()
                DataManager.fruits_data_lock.release()
            elif action == ModuleTransferAction.NAV:
                # write GPS data to .nav file
                logging.info(f"WRITING NAV DATA TO FILE")
                nav_path = tools.get_nav_path()
                nav_df = pd.DataFrame(data)
                is_first = not os.path.exists(nav_path)
                nav_df.to_csv(nav_path, header=is_first, index=False, mode='a')
        elif sender_module == ModulesEnum.Analysis:
            if action == ModuleTransferAction.FRUITS_DATA:
                logging.info(f"FRUIT DATA RECEIVED")
                DataManager.fruits_data_lock.acquire(blocking=True)
                if not data["fruit id"]:
                    DataManager.start_new_file()
                for k, v in data.items():
                    try:
                        DataManager.fruits_data[k] += v
                    except KeyError:
                        DataManager.fruits_data[k] = v
                DataManager.fruits_data_lock.release()
            elif action == ModuleTransferAction.IMU:
                # write IMU data to .imu file
                logging.info(f"WRITING NAV DATA TO FILE")
                imu_path = tools.get_imu_path()
                imu_df = pd.DataFrame(data)
                is_first = not os.path.exists(imu_path)
                imu_df.to_csv(imu_path, header=is_first)
        elif sender_module == ModulesEnum.GUI:
            if action == ModuleTransferAction.START_ACQUISITION:
                DataManager.previous_plot = DataManager.current_plot
                DataManager.current_plot = data["plot"]
                DataManager.current_row = data["row"]

                # update index and path for the new file
                plot_dir = os.listdir(tools.get_fruits_path(
                    plot=DataManager.current_plot,
                    row=DataManager.current_row,
                    get_dir=True)
                )
                path_indices = [tools.index_from_svo(f, as_int=True) for f in plot_dir if tools.is_svo(f)]
                print("PATH INDICES: ", path_indices)
                DataManager.current_index = 1 + max(path_indices, default=0)
                DataManager.current_path = tools.get_fruits_path(
                    plot=DataManager.current_plot,
                    index=DataManager.current_index,
                    row=DataManager.current_row
                )

            if action == ModuleTransferAction.STOP_ACQUISITION:
                today = datetime.now().strftime("%d%m%y")
                collected_data = {
                    "customer_code": [conf["customer code"]],
                    "plot_code": [DataManager.current_plot],
                    "scan_date": [today],
                    "row": [DataManager.current_row],
                    "file_index": [DataManager.current_index]
                }
                tmp_df = pd.DataFrame(data=collected_data, index=[0])
                DataManager.collected_df = pd.concat([DataManager.collected_df, tmp_df], axis=0).drop_duplicates()
                DataManager.collected_df.to_csv(data_conf["collected path"], mode="w", index=False, header=True)

    @staticmethod
    def write_fruits_data_locally(release=True):
        if DataManager.current_path:
            DataManager.fruits_data_lock.acquire(blocking=True)
            fruits_df = pd.DataFrame(data=DataManager.fruits_data)
            is_first = not os.path.exists(DataManager.current_path)
            fruits_df.to_csv(DataManager.current_path, sep=",", mode="a+", index=False, header=is_first)
            if release:
                DataManager.fruits_data_lock.release()

    @staticmethod
    def update_output():
        while not DataManager.shutdown_event.wait(data_conf["update interval"]):
            DataManager.write_fruits_data_locally()

    @staticmethod
    def internet_scan():
        while True:
            t0 = time.time()
            try:
                # get the upload speed in KB/s
                upload_in_kbps = speedtest.Speedtest().upload() / (1024 * 8)
                logging.info(f"INTERNET SCAN - START - UPLOAD SPEED = {upload_in_kbps} KB/s")
                # tools.s3_upload_previous_nav_log()
            except speedtest.SpeedtestException:
                logging.info(f"INTERNET SCAN - NO CONNECTION")
            finally:
                DataManager.scan_files(upload_timeout=data_conf["upload interval"] - 30)
                t1 = time.time()
                logging.info(f"INTERNET SCAN - END")
                next_execution_time = max(0.1, data_conf["upload interval"] - (t1 - t0))
                if DataManager.shutdown_event.wait(next_execution_time):
                    break
        logging.info("INTERNET SCAN - FINISHED")

    @staticmethod
    def scan_files(upload_timeout):
        logging.info("SCANNING FILES...")
        file_suffix = "feather" if data_conf["use feather"] else "csv"
        try:
            try:
                scan_csv_df = pd.read_csv(data_conf["scanner path"], dtype=str)
                # pull latest data from scan_df into scan.csv
                DataManager.scan_lock.acquire(blocking=True)
                scan_csv_df = pd.concat([scan_csv_df, DataManager.scan_df], axis=0)
                DataManager.scan_df = pd.DataFrame(data={"customer code": [], "plot code": [], "scan date": [], "filename": []})
                DataManager.scan_lock.release()
            except FileNotFoundError:
                # if scan.csv does not exist, copy scan_df
                DataManager.scan_lock.acquire(blocking=True)
                DataManager.scan_df.to_csv(data_conf["scanner path"], mode="w", index=False, header=True)
                DataManager.scan_lock.release()
                scan_csv_df = DataManager.scan_df

            removed_indices, removed_files = [], []
            logging.info(f"SCANNING {len(scan_csv_df)} FILES")

            # go through the files of every customer-plot-date separately
            plots_df_gr = scan_csv_df.groupby(["customer code", "plot code", "scan date"])
            for _, plot_df in plots_df_gr:
                if upload_timeout <= 0:
                    break
                t0 = time.time()

                customer_code = plot_df["customer code"].iloc[0]
                plot_code = plot_df["plot code"].iloc[0]
                scan_date = plot_df["scan date"].iloc[0]

                indices = [tools.index_from_fruits(filename) for filename in plot_df["filename"]]
                if upload_timeout <= 0.1:
                    logging.info(f"SCAN - UPLOAD TIMEOUT - STOP UPLOADING")
                    break
                # try to upload all files in current chunk. valid_indices are the
                success, valid_indices = tools.upload_to_s3(customer_code, plot_code, scan_date, indices,
                                                            upload_timeout)
                logging.info(f"SCAN - UPLOAD TIMEOUT STATUS - {upload_timeout} SECONDS")
                if not success:
                    break
                if not valid_indices:
                    continue
                try:
                    response = tools.send_request_to_server(customer_code, plot_code, scan_date, valid_indices)
                    logging.info(f"REQUEST (TRYING) - CUSTOMER: {customer_code}, PLOT: {plot_code}, "
                                 f"INDICES: {valid_indices}")
                    if not response.ok:
                        logging.error("DATA MANAGER - REQUEST FAILED")
                        continue
                    else:
                        remove_from_valid = [(customer_code, plot_code, scan_date, f"fruits_{i}.{file_suffix}")
                                             for i in valid_indices]
                        removed_files += remove_from_valid
                        logging.info("REQUEST SUCCESS")
                except RequestException:
                    logging.error("REQUEST FAILED")
                    continue
                finally:
                    t1 = time.time()
                    upload_timeout -= t1 - t0

            if not scan_csv_df.empty:
                def filter_removed(row):
                    c, p, sd, r, f = row["customer code"], row["plot code"], str(row["scan date"]), row["row"], row[
                        "filename"]
                    return (c, p, sd, r, f) not in removed_files

                scan_csv_df = scan_csv_df.loc[scan_csv_df.apply(func=filter_removed, axis=1)]
                scan_csv_df.drop_duplicates(inplace=True)
            scan_csv_df.to_csv(data_conf["scanner path"], mode="w", index=False, header=True)
        except Exception:
            logging.exception("SCANNING ERROR")
        finally:
            logging.info("SCAN FINISHED")

    @staticmethod
    def scan_analyzed(scan_timeout):
        timeout_time = time.time() + scan_timeout
        analyzed_csv_df = None
        try:
            while time.time() < timeout_time:
                try:
                    analyzed_csv_df = pd.read_csv(data_conf["scanner path"], dtype=str)
                    break
                except PermissionError:
                    time.sleep(5)
            if not analyzed_csv_df:
                return
            # pull latest data from scan_df into scan.csv
            analyzed_csv_df = pd.concat([analyzed_csv_df, DataManager.analyzed_df], axis=0)
            DataManager.analyzed_df = pd.DataFrame(data={"customer code": [], "plot code": [], "scan date": [], "filename": []})
            DataManager.scan_lock.release()
        except FileNotFoundError:
            # if scan.csv does not exist, copy scan_df
            DataManager.scan_lock.acquire(blocking=True)
            DataManager.scan_df.to_csv(data_conf["scanner path"], mode="w", index=False, header=True)
            DataManager.scan_lock.release()
            scan_csv_df = DataManager.scan_df
