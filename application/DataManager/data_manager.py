import os.path
import threading
import traceback
from builtins import staticmethod
import time
from datetime import datetime
import signal
import logging
import boto3
import pandas as pd
import requests
from boto3.exceptions import S3UploadFailedError
from botocore.exceptions import EndpointConnectionError
from cupy._core._accelerator import accelerator_type
from requests.exceptions import RequestException
from application.utils.settings import data_conf, conf
from application.utils.module_wrapper import ModulesEnum, Module, ModuleTransferAction
import application.utils.tools as tools
import speedtest


class DataManager(Module):
    previous_plot, current_plot = data_conf.global_polygon, data_conf.global_polygon
    current_row = -1
    current_path, current_index = None, -1
    fruits_data = dict()
    fruits_data_lock, scan_lock, analyzed_lock = threading.Lock(), threading.Lock(), threading.Lock()
    s3_client = None
    update_output_thread, internet_scan_thread = None, None
    scan_df = pd.DataFrame(data={"customer_code": [], "plot_code": [], "scan_date": [], "row": [], "filename": []})
    collected_df = pd.DataFrame(
        data={"customer_code": [], "plot_code": [], "scan_date": [], "row": [], "folder_index": []})
    uploaded_df = pd.DataFrame(
        data={"customer_code": [], "plot_code": [], "scan_date": [], "row": [], "folder_index": []})

    @staticmethod
    def init_module(qu, main_pid, module_name, communication_queue):

        def try_read(path):
            try:
                df = pd.read_csv(path, dtype=str)
            except FileNotFoundError:
                df = pd.DataFrame(
                    data={"customer_code": [], "plot_code": [], "scan_date": [], "row": [], "folder_index": []})
            return df

        super(DataManager, DataManager).init_module(qu, main_pid, module_name, communication_queue)
        super(DataManager, DataManager).set_signals(DataManager.shutdown, DataManager.receive_data)

        DataManager.s3_client = boto3.client("s3")
        # DataManager.update_output_thread = threading.Thread(target=DataManager.update_output, daemon=True)
        DataManager.internet_scan_thread = threading.Thread(target=DataManager.internet_scan, daemon=True)

        DataManager.collected_df = try_read(data_conf.collected_path)
        DataManager.uploaded_df = try_read(data_conf.uploaded_path)

        # DataManager.update_output_thread.start()
        DataManager.internet_scan_thread.start()
        # DataManager.update_output_thread.join()
        DataManager.internet_scan_thread.join()

    @staticmethod
    def receive_data(sig, frame):
        data, sender_module = DataManager.qu.get()
        action, data = data["action"], data["data"]
        if sender_module == ModulesEnum.GPS:
            if action == ModuleTransferAction.NAV:
                # write GPS data to .nav file
                logging.info(f"WRITING NAV DATA TO FILE")
                nav_path = tools.get_nav_path()
                nav_df = pd.DataFrame(data)
                is_first = not os.path.exists(nav_path)
                nav_df.to_csv(nav_path, header=is_first, index=False, mode='a')
        elif sender_module == ModulesEnum.Analysis:
            if action == ModuleTransferAction.FRUITS_DATA:
                logging.info(f"FRUIT DATA RECEIVED")
                with DataManager.fruits_data_lock:
                    for k, v in data.items():
                        try:
                            DataManager.fruits_data[k] += v
                        except KeyError:
                            DataManager.fruits_data[k] = v
            elif action == ModuleTransferAction.IMU:
                # write IMU data to .imu file
                logging.info(f"WRITING NAV DATA TO FILE")
                imu_path = tools.get_imu_path()
                imu_df = pd.DataFrame(data)
                is_first = not os.path.exists(imu_path)
                imu_df.to_csv(imu_path, header=is_first)
            elif action == ModuleTransferAction.ANALYZED_DATA:
                customer_code, plot_code, scan_date, row, folder_index = list(data["row"])
                is_success = data['status']
                logging.info(f"ANALYZED DATA ARRIVED: "
                             f"{data_conf.output_path}, {customer_code}, {plot_code}, {scan_date}, {row}")

                ext = "feather" if data_conf.use_feather else "csv"
                if is_success:
                    analyzed_path = os.path.join(data_conf.output_path,
                                                 customer_code,
                                                 plot_code,
                                                 str(scan_date),
                                                 f"row_{row}")


                    tracks, tracks_headers = data["tracks"], data["tracks_headers"]
                    tracks_path = os.path.join(analyzed_path, str(folder_index), f"{data_conf.tracks}.{ext}")
                    try:
                        tracks_df = pd.DataFrame(data=tracks, columns=tracks_headers)
                    except ValueError:
                        tracks_df = pd.DataFrame(columns=tracks_headers)

                    alignment, alignment_headers = data["alignment"], data["alignment_headers"]
                    alignment_path = os.path.join(analyzed_path, str(folder_index), f"{data_conf.alignment}.{ext}")

                    try:
                        alignment_df = pd.DataFrame(data=alignment, columns=alignment_headers)
                    except ValueError:
                        alignment_df = pd.DataFrame(columns=alignment_headers)

                    if data_conf.use_feather:
                        tracks_df.to_feather(tracks_path)
                        alignment_df.to_feather(alignment_path)
                    else:
                        tracks_df.to_csv(tracks_path, index=False, header=True)
                        alignment_df.to_csv(alignment_path, index=False, header=True)

                status = 'success' if is_success else 'failed'
                analyzed_data = {
                    "customer_code": [customer_code],
                    "plot_code": [plot_code],
                    "scan_date": [str(scan_date)],
                    "row": [str(int(row))],
                    "folder_index": [str(int(folder_index))],
                    "status": [status],
                    "ext": [ext]
                }

                analyzed_df = pd.DataFrame(data=analyzed_data, index=[0])
                is_first = not os.path.exists(data_conf.analyzed_path)
                with DataManager.analyzed_lock:
                    analyzed_df.to_csv(data_conf.analyzed_path, header=is_first, index=False, mode="a+")
        elif sender_module == ModulesEnum.Acquisition:
            if action == ModuleTransferAction.START_ACQUISITION:
                DataManager.previous_plot = DataManager.current_plot
                DataManager.current_plot = data["plot"]
                DataManager.current_row = data["row"]
                DataManager.current_index = data["folder_index"]

                DataManager.current_path = tools.get_path(
                    plot=DataManager.current_plot,
                    row=DataManager.current_row,
                    index=DataManager.current_index,
                    get_index_dir=True
                )
            elif action == ModuleTransferAction.STOP_ACQUISITION:
                if data_conf.use_feather:
                    filename_csv = f"{data_conf.jaized_timestamps}.csv"
                    filename_feather = f"{data_conf.jaized_timestamps}.feather"
                    jaized_timestamps_csv_path = os.path.join(DataManager.current_path, filename_csv)
                    jaized_timestamps_feather_path = os.path.join(DataManager.current_path, filename_feather)
                    pd.read_csv(jaized_timestamps_csv_path).to_feather(jaized_timestamps_feather_path)

                today = datetime.now().strftime("%d%m%y")
                collected_data = {
                    "customer_code": [conf.customer_code],
                    "plot_code": [DataManager.current_plot],
                    "scan_date": [today],
                    "row": [str(int(DataManager.current_row))],
                    "folder_index": [str(int(DataManager.current_index))]
                }
                tmp_df = pd.DataFrame(data=collected_data, index=[0])
                DataManager.collected_df = pd.concat([DataManager.collected_df, tmp_df], axis=0).drop_duplicates()
                DataManager.collected_df.to_csv(data_conf.collected_path, mode="w", index=False, header=True)
            elif action == ModuleTransferAction.JAIZED_TIMESTAMPS:
                input_length = len(data["JAI_frame_number"])
                data["row"] = [DataManager.current_row] * input_length
                data["folder_index"] = [DataManager.current_index] * input_length

                jaized_timestamp_path = os.path.join(DataManager.current_path, f"{data_conf.jaized_timestamps}.csv")
                jaized_timestamp_total_log_path = tools.get_jaized_timestamps_path()
                jaized_timestamp_log_df = pd.DataFrame(data)

                is_first = not os.path.exists(jaized_timestamp_path)
                jaized_timestamp_log_df.to_csv(jaized_timestamp_path, mode='a+', header=is_first, index=False)
                is_first = not os.path.exists(jaized_timestamp_total_log_path)
                jaized_timestamp_log_df.to_csv(jaized_timestamp_total_log_path, mode='a+', header=is_first, index=False)

    @staticmethod
    def update_output():
        while not DataManager.shutdown_event.wait(data_conf.update_interval):
            pass

    @staticmethod
    def internet_scan():
        while True:
            t0 = time.time()
            DataManager.scan_analyzed(scan_timeout=data_conf.upload_interval - 30)
            t1 = time.time()
            logging.info(f"INTERNET SCAN - END")
            print(f"INTERNET SCAN - END")
            next_execution_time = max(0.1, data_conf.upload_interval - (t1 - t0))
            if DataManager.shutdown_event.wait(next_execution_time):
                break
        logging.info("INTERNET SCAN - FINISHED")

    @staticmethod
    def scan_analyzed(scan_timeout):

        logging.info("START SCANNING ANALYZED FILES")
        print("START SCANNING ANALYZED FILES")
        t_scan_start = time.time()
        try:
            upload_speed_in_bps = speedtest.Speedtest().upload()
            upload_speed_in_kbps = upload_speed_in_bps / (1024 * 8)
            logging.info(f"INTERNET UPLOAD SPEED - {upload_speed_in_kbps} KB/s")
            print(f"INTERNET UPLOAD SPEED - {upload_speed_in_kbps} KB/s")
            if upload_speed_in_kbps < 10:
                raise speedtest.SpeedtestException
        except speedtest.SpeedtestException:
            logging.info("NO INTERNET CONNECTION")
            print("NO INTERNET CONNECTION")
            return
        except Exception:
            logging.exception("unknown handled exception: ")
            return

        def upload_analyzed(timeout_before, analyzed_group):

            print("uploading group")
            print(analyzed_group)

            def get_data_size(tracks, alignment, jaized_timestamps):
                tracks_size = os.path.getsize(tracks)
                alignment_size = os.path.getsize(alignment)
                timestamps_size = os.path.getsize(jaized_timestamps)
                return (tracks_size + alignment_size + timestamps_size) / 1024

            t0 = time.time()
            _customer_code = analyzed_group["customer_code"].iloc[0]
            _plot_code = analyzed_group["plot_code"].iloc[0]
            _scan_date = str(analyzed_group["scan_date"].iloc[0])
            _uploaded_indices = {}
            for _, analyzed_row in analyzed_group.iterrows():
                folder_index = str(analyzed_row["folder_index"])
                row = f"row_{analyzed_row['row']}"
                folder_name = os.path.join(_customer_code, _plot_code, _scan_date, row, folder_index)
                folder_path = os.path.join(data_conf.output_path, folder_name)
                ext = "csv"
                # TODO: modify the 'collected', 'analyzed' and 'uploaded' to contain the file type (csv / feather)
                tracks_path = os.path.join(folder_path, f"{data_conf.tracks}.{ext}")
                tracks_s3_path = os.path.join(data_conf.upload_prefix, folder_name, f"{data_conf.tracks}.{ext}")

                alignment_path = os.path.join(folder_path, f"{data_conf.alignment}.{ext}")
                alignment_s3_path = os.path.join(data_conf.upload_prefix, folder_name, f"{data_conf.alignment}.{ext}")

                timestamps_path = os.path.join(folder_path, f"{data_conf.jaized_timestamps}.{ext}")
                if not os.path.exists(timestamps_path):
                    timestamps_path_log = timestamps_path.replace('csv', 'log')
                    os.rename(timestamps_path_log, timestamps_path)
                    print("renamed ", timestamps_path)
                timestamps_s3_path = os.path.join(data_conf.upload_prefix, folder_name,
                                                  f"{data_conf.jaized_timestamps}.{ext}")

                data_size_in_kb = get_data_size(tracks_path, alignment_path, timestamps_path)
                if data_size_in_kb >= upload_speed_in_kbps * timeout:
                    continue
                try:
                    DataManager.s3_client.upload_file(tracks_path, data_conf.upload_bucket_name, tracks_s3_path)
                    DataManager.s3_client.upload_file(alignment_path, data_conf.upload_bucket_name, alignment_s3_path)
                    DataManager.s3_client.upload_file(timestamps_path, data_conf.upload_bucket_name, timestamps_s3_path)
                    try:
                        _uploaded_indices[row].append(folder_index)
                    except KeyError:
                        _uploaded_indices[row] = [folder_index]
                except TimeoutError:
                    print("timeout error")
                    break
                except FileNotFoundError:
                    logging.warning(f"UPLOAD TO S3 - MISSING INDEX - {folder_name}")
                    print(f"UPLOAD TO S3 - MISSING INDEX - {folder_name}")
                    break
                except EndpointConnectionError:
                    logging.warning(f"UPLOAD TO S3 - FAILED DUE TO INTERNET CONNECTION  - {folder_name}")
                    print(f"UPLOAD TO S3 - FAILED DUE TO INTERNET CONNECTION  - {folder_name}")
                    break
                except S3UploadFailedError:
                    logging.warning(f"UPLOAD TO S3 - FAILED DUE TO S3 RELATED PROBLEM  - {folder_name}")
                    print(f"UPLOAD TO S3 - FAILED DUE TO S3 RELATED PROBLEM  - {folder_name}")
                    break
                except Exception:
                    logging.exception(f"UPLOAD TO S3 - FAILED DUE TO AN ERROR - {folder_name}")
                    traceback.print_exc()
                    break

            t_delta = time.time() - t0
            timeout_after = timeout_before - t_delta
            return _customer_code, _plot_code, _scan_date, _uploaded_indices, timeout_after

        def send_request(timeout_before, _customer_code, _plot_code, _scan_date, _uploaded_indices):
            if not uploaded_indices:
                return timeout_before, False

            t0 = time.time()
            headers = {"Content-Type": "application/json; charset=utf-8", 'Accept': 'text/plain'}
            request_data = {
                "customer_code": _customer_code,
                "plot_code": _plot_code,
                "scan_date": _scan_date,
                "indices": _uploaded_indices,
                "output dir": os.path.join(_customer_code, _plot_code, _scan_date),
                "output types": ['FSI']
            }

            print("request sent")
            response = requests.post(data_conf.service_endpoint, json=request_data, headers=headers, timeout=timeout)
            if response.ok:
                print("request success")
                _uploaded_dict = {"customer_code": [], "plot_code": [], "scan_date": [], "row": [], "folder_index": []}
                for row_name, indices in uploaded_indices.items():
                    row_number = row_name.split('_')[1]
                    for index in indices:
                        _uploaded_dict["customer_code"].append(customer_code)
                        _uploaded_dict["plot_code"].append(plot_code)
                        _uploaded_dict["scan_date"].append(scan_date)
                        _uploaded_dict["row"].append(row_number)
                        _uploaded_dict["folder_index"].append(index)
                is_first = not os.path.exists(data_conf.uploaded_path)
                pd.DataFrame(_uploaded_dict).to_csv(data_conf.uploaded_path, mode='a+', index=False, header=is_first)

            t_delta = time.time() - t0
            timeout_after = timeout_before - t_delta
            return timeout_after, response.ok

        analyzed_csv_df = None
        future_timeout = t_scan_start + scan_timeout
        while time.time() < future_timeout:
            try:
                analyzed_csv_df = pd.read_csv(data_conf.analyzed_path, dtype=str)
                # TODO: READ 'uploaded' AND FILTER OUT THOSE FILES
                break
            except PermissionError:
                time.sleep(5)
            except FileNotFoundError:
                return

        if analyzed_csv_df is None:
            return

        uploaded_csv_df = None
        try:
            uploaded_csv_df = pd.read_csv(data_conf.uploaded_path, dtype=str)
        except FileNotFoundError:
            pass

        t_delta = time.time() - t_scan_start
        if uploaded_csv_df is not None:
            print("ANALYZED:\n", analyzed_csv_df)
            print("UPLOADED:\n", uploaded_csv_df)
            analyzed_not_uploaded = pd.merge(analyzed_csv_df, uploaded_csv_df, how='left', indicator=True)
            not_uploaded = analyzed_not_uploaded['_merge'] == 'left_only'
            analyzed_not_uploaded = analyzed_not_uploaded.loc[not_uploaded, analyzed_not_uploaded.columns != '_merge']
            print("NOT UPLOADED:\n", analyzed_not_uploaded)
        else:
            analyzed_not_uploaded = analyzed_csv_df

        analyzed_groups = analyzed_not_uploaded.groupby(["customer_code", "plot_code", "scan_date"])
        timeout = scan_timeout - t_delta

        for _, analyzed_gr in analyzed_groups:
            customer_code, plot_code, scan_date, uploaded_indices, timeout = upload_analyzed(timeout, analyzed_gr)
            timeout, response_ok = send_request(timeout, customer_code, plot_code, scan_date, uploaded_indices)

        # analyzed_groups = analyzed_csv_df.groupby(["customer code", "plot code", "scan date"])
        # for _, analyzed_group in analyzed_groups:
        #     group_rows = analyzed_group.groupby(["row"])
        #     for _, gr in group_rows:
        #         indices = gr["folder_index"].to_list()
        #     for _, r in group.iterrows():
        #         row, folder_index = r["row"], r["folder_index"]
        #
        # if scan.csv does not exist, copy scan_df
        with DataManager.scan_lock:
            DataManager.scan_df.to_csv(data_conf.scanner_path, mode="w", index=False, header=True)
        scan_csv_df = DataManager.scan_df
