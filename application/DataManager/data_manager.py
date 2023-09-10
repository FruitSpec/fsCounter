import os.path
import shutil
import threading
import traceback

from jtop import jtop, JtopException
from builtins import staticmethod
import time
from datetime import datetime
import psutil
import logging
import boto3
from botocore.config import Config
import pandas as pd
import requests
from boto3.exceptions import S3UploadFailedError
from botocore.exceptions import EndpointConnectionError
from application.utils.settings import data_conf, conf, consts
from application.utils.module_wrapper import ModulesEnum, Module, ModuleTransferAction
import application.utils.tools as tools
from application.utils.settings import set_logger
import speedtest


class DataManager(Module):
    previous_plot, current_plot = consts.global_polygon, consts.global_polygon
    is_in_plot = False
    current_row = -1
    current_path, current_index = None, -1
    fruits_data = dict()
    s3_client = None
    internet_scan_thread, receive_data_thread = None, None
    nav_df = pd.DataFrame(data={"timestamp": [], "latitude": [], "longitude": [], "plot": []})
    collected_df = pd.DataFrame(
        data={"customer_code": [], "plot_code": [], "scan_date": [], "row": [], "folder_index": [], "ext": []})

    @staticmethod
    def init_module(in_qu, out_qu, main_pid, module_name, communication_queue, notify_on_death, death_action):

        def try_read(path):
            try:
                df = pd.read_csv(path, dtype=str)
            except FileNotFoundError:
                df = pd.DataFrame(
                    data={"customer_code": [], "plot_code": [], "scan_date": [], "row": [], "folder_index": [],
                          "ext": []})
            return df

        super(DataManager, DataManager).init_module(in_qu, out_qu, main_pid, module_name, communication_queue,
                                                    notify_on_death, death_action)
        super(DataManager, DataManager).set_signals(DataManager.shutdown)

        DataManager.s3_client = boto3.client('s3', config=Config(retries={"total_max_attempts": 1}))

        DataManager.receive_data_thread = threading.Thread(target=DataManager.receive_data, daemon=True)
        DataManager.internet_scan_thread = threading.Thread(target=DataManager.internet_scan, daemon=True)

        DataManager.collected_df = try_read(data_conf.collected_path)

        DataManager.receive_data_thread.start()
        DataManager.internet_scan_thread.start()

        if data_conf.network_log:
            DataManager.network_monitor_thread = threading.Thread(target=DataManager.network_monitor, daemon=True)
            DataManager.network_monitor_thread.start()

        if data_conf.jtop_log:
            jtop_t = threading.Thread(target=DataManager.jtop_logger, daemon=True)
            jtop_t.start()

        DataManager.internet_scan_thread.join()

    @staticmethod
    def receive_data():

        def jaized_timestamps():
            try:
                input_length = len(data[consts.JAI_frame_number])
                data["row"] = [DataManager.current_row] * input_length
                data["folder_index"] = [DataManager.current_index] * input_length

                jaized_timestamp_path = os.path.join(DataManager.current_path, f"{consts.jaized_timestamps}.csv")
                jaized_timestamp_total_log_path = tools.get_file_path(tools.FileTypes.jaized_timestamps)
                jaized_timestamp_log_df = pd.DataFrame(data)

                _is_first = not os.path.exists(jaized_timestamp_path)
                jaized_timestamp_log_df.to_csv(jaized_timestamp_path, mode='a+', header=_is_first, index=False)

                _is_first = not os.path.exists(jaized_timestamp_total_log_path)
                jaized_timestamp_log_df.to_csv(jaized_timestamp_total_log_path, mode='a+', header=_is_first, index=False)
            except:
                tools.log("JAIZED TIMESTAMP ERROR", logging.ERROR, exc_info=True)

        def start_acquisition():
            if DataManager.is_in_plot:
                tools.log("START_ACQUISITION RECEIVED WHILE IN PLOT", log_level=logging.WARNING)
            DataManager.is_in_plot = True
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

        def stop_acquisition():
            if not DataManager.is_in_plot:
                return

            DataManager.is_in_plot = False

            filename_csv = f"{consts.jaized_timestamps}.csv"
            jaized_timestamps_csv_path = os.path.join(DataManager.current_path, filename_csv)
            jaized_timestamp_log_df = pd.read_csv(jaized_timestamps_csv_path).sort_values(by=consts.JAI_frame_number)

            if data_conf.use_feather:
                filename_feather = f"{consts.jaized_timestamps}.feather"
                jaized_timestamps_feather_path = os.path.join(DataManager.current_path, filename_feather)
                jaized_timestamp_log_df.to_feather(jaized_timestamps_feather_path)
            else:
                jaized_timestamp_log_df.to_csv(jaized_timestamps_csv_path, header=True, index=False)

            disk_occupancy = psutil.disk_usage("/").percent
            tools.log(f"DISK OCCUPANCY {disk_occupancy}%")
            tools.log(f"DISK OCCUPANCY THRESHOLD {data_conf.max_disk_occupancy}%")

            if conf.collect_data:
                today = datetime.now().strftime("%d%m%y")
                ext = "feather" if data_conf.use_feather else "csv"
                collected_data = {
                    "customer_code": [conf.customer_code],
                    "plot_code": [DataManager.current_plot],
                    "scan_date": [today],
                    "row": [str(int(DataManager.current_row))],
                    "folder_index": [str(int(DataManager.current_index))],
                    "ext": [ext]
                }

                tmp_collected_df = pd.DataFrame(data=collected_data, index=[0])
                _is_first = not os.path.exists(data_conf.collected_path)
                tmp_collected_df.to_csv(data_conf.collected_path, mode="a+", index=False, header=_is_first)

            if disk_occupancy > data_conf.max_disk_occupancy:
                time.sleep(1)
                tools.log("RESTARTING TO PERFORM DISK CLEANUP")
                DataManager.send_data(ModuleTransferAction.RESTART_APP, None, ModulesEnum.Main)

        while not DataManager.shutdown_event.is_set():
            data, sender_module = DataManager.in_qu.get()
            action, data = data["action"], data["data"]
            if sender_module == ModulesEnum.GPS:
                if action == ModuleTransferAction.NAV:
                    # write GPS data to .nav file
                    tools.log(f"WRITING NAV DATA TO FILE", log_option=tools.LogOptions.LOG)
                    new_nav_df = pd.DataFrame(data)
                    DataManager.nav_df = pd.concat([DataManager.nav_df, new_nav_df], axis=0)
                    nav_path = tools.get_file_path(tools.FileTypes.nav)
                    is_first = not os.path.exists(nav_path)
                    new_nav_df.to_csv(nav_path, header=is_first, index=False, mode='a+')
                elif action == ModuleTransferAction.JAIZED_TIMESTAMPS:
                    jaized_timestamps()
            elif sender_module == ModulesEnum.Analysis:
                if action == ModuleTransferAction.FRUITS_DATA:
                    tools.log(f"FRUIT DATA RECEIVED")
                    for k, v in data.items():
                        try:
                            DataManager.fruits_data[k] += v
                        except KeyError:
                            DataManager.fruits_data[k] = v
                elif action == ModuleTransferAction.ANALYZED_DATA:
                    def write_locally(_name):
                        _data_key = _name
                        _headers_key = f"{_name}_header"
                        _filename = _name
                        nonlocal ext, data, analyzed_path, folder_index
                        _df_data, _df_headers = data[_data_key], data[_headers_key]
                        _file_path = os.path.join(analyzed_path, str(folder_index), f"{_filename}.{ext}")
                        try:
                            _df = pd.DataFrame(data=_df_data, columns=_df_headers)
                        except ValueError:
                            _df = pd.DataFrame(columns=_df_headers)

                        if data_conf.use_feather:
                            _df.to_feather(_file_path)
                        else:
                            _df.to_csv(_file_path, index=False, header=True)

                    customer_code, plot_code, scan_date, row, folder_index, ext = list(data["row"])
                    is_success = data["status"]
                    tools.log(f"ANALYZED DATA ARRIVED: "
                                 f"{data_conf.output_path}, {customer_code}, {plot_code}, {scan_date}, {row}")

                    customer_code, plot_code, scan_date, row, folder_index, ext = list(data["row"])
                    is_success = data["status"]
                    tools.log(f"ANALYZED DATA ARRIVED: "
                                 f"{data_conf.output_path}, {customer_code}, {plot_code}, {scan_date}, {row}")

                    if is_success:
                        analyzed_path = os.path.join(
                            data_conf.output_path, customer_code, plot_code,
                            str(scan_date), f"row_{row}"
                        )

                        write_locally(consts.tracks)
                        write_locally(consts.alignment)
                        if conf.crop == consts.citrus:
                            write_locally(consts.jai_translation)

                    status = consts.success if is_success else consts.failed
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
                    analyzed_df.to_csv(data_conf.analyzed_path, header=is_first, index=False, mode="a+")
            elif sender_module == ModulesEnum.Acquisition:
                if action == ModuleTransferAction.START_ACQUISITION:
                    start_acquisition()
                elif action == ModuleTransferAction.STOP_ACQUISITION:
                    stop_acquisition()
                elif action == ModuleTransferAction.ACQUISITION_CRASH:
                    tools.log("HANDLING ACQUISITION CRASH")

                    stop_acquisition()

                    # copy syslog to crash dir
                    if data_conf.crash_syslog:
                        now = datetime.now().strftime("%d%m%y_%H%M%S")
                        crash_syslog_filename = f"{consts.syslog}_{now}.{consts.log_extension}"
                        crash_syslog_path = os.path.join(consts.crash_dir, crash_syslog_filename)
                        shutil.copyfile(consts.syslog_path, crash_syslog_path)

                    tools.log("ACQUISITION CRASH HANDLING DONE")
            elif sender_module == ModulesEnum.Main:
                if action == ModuleTransferAction.MONITOR:
                    DataManager.send_data(
                        action=ModuleTransferAction.MONITOR,
                        data=None,
                        receiver=ModulesEnum.Main,
                        log_option=tools.LogOptions.NONE
                    )
                elif action == ModuleTransferAction.SET_LOGGER:
                    set_logger()

    @staticmethod
    def update_output():
        while not DataManager.shutdown_event.wait(data_conf.update_interval):
            pass

    @staticmethod
    def internet_scan():
        last_nav_upload = time.time()
        while not DataManager.shutdown_event.is_set():
            upload_speed_in_kbps = 0
            try:
                upload_speed_in_bps = speedtest.Speedtest().upload()
                upload_speed_in_kbps = upload_speed_in_bps / (1024 * 8)
                tools.log(f"INTERNET UPLOAD SPEED - {upload_speed_in_kbps} KB/s")
            except speedtest.SpeedtestException:
                tools.log("NO INTERNET CONNECTION")
            except Exception:
                tools.log("UNKNOWN HANDLED EXCEPTION: ", logging.ERROR, exc_info=True)
            t0 = time.time()
            if upload_speed_in_kbps > 10:
                timeout = data_conf.upload_interval - 30

                # upload nav file once every {nav_upload_interval} seconds
                # if not uploaded successfully, keep trying every 5 minutes
                now = time.time()
                if (now - last_nav_upload) > data_conf.nav_upload_interval:
                    is_successful, timeout = DataManager.upload_today_files(upload_speed_in_kbps, timeout=timeout)
                    if is_successful:
                        last_nav_upload = time.time()
                else:
                    print()
                timeout = DataManager.upload_old_files(upload_speed_in_kbps, timeout=timeout)
                DataManager.scan_analyzed(upload_speed_in_kbps, timeout)
                tools.log(f"INTERNET SCAN - END")
            t1 = time.time()
            next_execution_time = max(10, data_conf.upload_interval - (t1 - t0))
            if DataManager.shutdown_event.wait(next_execution_time):
                break

        tools.log("INTERNET SCAN - FINISHED")

    @staticmethod
    def upload_to_s3(path, s3_path, upload_speed_in_kbps, timeout, extension=None):
        filename = os.path.basename(path)
        try:
            _size_in_kb = os.path.getsize(path) / 1024
            if _size_in_kb >= upload_speed_in_kbps * timeout:
                tools.log(f"UPLOAD {filename} - NOT ENOUGH TIME LEFT")
                return
            DataManager.s3_client.upload_file(path, data_conf.upload_bucket_name, s3_path)
            if extension:
                path_uploaded = path.replace(f".{extension}", f"_uploaded.{extension}")
                os.rename(path, path_uploaded)
            tools.log(f"UPLOAD {filename} TO S3 - SUCCESS")
            return True
        except FileNotFoundError:
            tools.log(f"UPLOAD {filename} - FILE NOT EXIST")
            return False
        except EndpointConnectionError:
            tools.log(f"UPLOAD {filename} TO S3 - FAILED DUE TO INTERNET CONNECTION", logging.WARNING)
            return False
        except S3UploadFailedError:
            tools.log(f"UPLOAD {filename} TO S3 - FAILED DUE TO S3 RELATED PROBLEM", logging.WARNING)
            return False
        except Exception:
            tools.log(f"UPLOAD {filename} TO S3 - FAILED DUE TO AN ERROR - {path}", logging.ERROR, exc_info=True)
            return False

    @staticmethod
    def upload_today_files(upload_speed_in_kbps, timeout=10):

        tools.log("UPLOADING TODAY FILES")

        nav_path, nav_s3_path = tools.get_file_path(
            tools.FileTypes.nav,
            with_s3_path=True,
            s3_folder_name=consts.s3_nav_folder
        )

        log_path, log_s3_path = tools.get_file_path(
            tools.FileTypes.log,
            with_s3_path=True,
            s3_folder_name=consts.s3_log_folder
        )
        jzts_path, jzts_s3_path = tools.get_file_path(
            tools.FileTypes.jaized_timestamps,
            with_s3_path=True,
            s3_folder_name=consts.s3_jaized_folder
        )

        t0 = time.time()

        nav_is_successful = DataManager.upload_to_s3(nav_path, nav_s3_path, upload_speed_in_kbps, timeout)
        log_is_successful = DataManager.upload_to_s3(log_path, log_s3_path, upload_speed_in_kbps, timeout)
        jzts_is_successful = DataManager.upload_to_s3(jzts_path, jzts_s3_path, upload_speed_in_kbps, timeout)

        is_successful = nav_is_successful and log_is_successful and jzts_is_successful

        t1 = time.time()
        timeout = max(timeout - (t1 - t0), 10)

        return is_successful, timeout

    @staticmethod
    def upload_old_files(upload_speed_in_kbps, timeout=10):

        tools.log("UPLOADING OLD FILES")

        old_nav_paths = tools.get_old_file_paths(tools.FileTypes.nav)
        old_log_paths = tools.get_old_file_paths(tools.FileTypes.log)
        old_jaized_timestamps_paths = tools.get_old_file_paths(tools.FileTypes.jaized_timestamps)

        old_all = [(old_nav_paths, consts.nav_extension)] + \
                  [(old_log_paths, consts.log_extension)] + \
                  [(old_jaized_timestamps_paths, consts.log_extension)]

        for old_list in old_all:
            old_paths, ext = old_list
            for local_path, s3_path in old_paths:
                t0 = time.time()
                tools.log(f"TRYING TO UPLOAD {os.path.basename(local_path)}")
                DataManager.upload_to_s3(local_path, s3_path, upload_speed_in_kbps, timeout, extension=ext)
                t1 = time.time()
                timeout = max(timeout - (t1 - t0), 3)

        return max(timeout, 10)

    @staticmethod
    def scan_analyzed(upload_speed_in_kbps, scan_timeout):

        tools.log("START SCANNING ANALYZED FILES")
        t_scan_start = time.time()

        def upload_analyzed(timeout_before, analyzed_group):
            def get_data_size(tracks, alignment, jaized_timestamps):
                tracks_size = os.path.getsize(tracks)
                alignment_size = os.path.getsize(alignment)
                timestamps_size = os.path.getsize(jaized_timestamps)
                return (tracks_size + alignment_size + timestamps_size) / 1024

            def add_to_dict(d, k, v):
                try:
                    d[k].append(v)
                except KeyError:
                    d[k] = [v]

            t0 = time.time()
            _customer_code = analyzed_group["customer_code"].iloc[0]
            _plot_code = analyzed_group["plot_code"].iloc[0]
            _scan_date = str(analyzed_group["scan_date"].iloc[0])
            _uploaded_indices = {}
            _uploaded_extensions = {}
            _failed_indices = {}
            for _, analyzed_row in analyzed_group.iterrows():
                folder_index = str(analyzed_row["folder_index"])
                row = f"row_{analyzed_row['row']}"
                folder_name = os.path.join(_customer_code, _plot_code, _scan_date, row, folder_index)
                folder_path = os.path.join(data_conf.output_path, folder_name)
                ext = "csv"

                tracks_path = os.path.join(folder_path, f"{consts.tracks}.{ext}")
                tracks_s3_path = tools.s3_path_join(folder_name, f"{consts.tracks}.{ext}")

                alignment_path = os.path.join(folder_path, f"{consts.alignment}.{ext}")
                alignment_s3_path = tools.s3_path_join(folder_name, f"{consts.alignment}.{ext}")

                timestamps_path = os.path.join(folder_path, f"{consts.jaized_timestamps}.{ext}")
                timestamps_s3_path = tools.s3_path_join(folder_name, f"{consts.jaized_timestamps}.{ext}")

                try:
                    data_size_in_kb = get_data_size(tracks_path, alignment_path, timestamps_path)
                    tools.log(f"TRYING TO UPLOAD {folder_name}")
                    if data_size_in_kb >= upload_speed_in_kbps * timeout:
                        tools.log(f"UPLOAD {folder_name} - NOT ENOUGH TIME LEFT")
                        continue
                    DataManager.s3_client.upload_file(tracks_path, data_conf.upload_bucket_name, tracks_s3_path)
                    DataManager.s3_client.upload_file(alignment_path, data_conf.upload_bucket_name, alignment_s3_path)
                    DataManager.s3_client.upload_file(timestamps_path, data_conf.upload_bucket_name, timestamps_s3_path)
                    add_to_dict(_uploaded_indices, row, folder_index)
                    add_to_dict(_uploaded_extensions, row, ext)
                    tools.log(f"UPLOAD {folder_name} - SUCCESS")
                except TimeoutError:
                    tools.log("timeout error")
                    break
                except FileNotFoundError:
                    tools.log(f"UPLOAD TO S3 - MISSING INDEX - {folder_name} - MARKED AS FAILED", logging.WARNING)
                    add_to_dict(_failed_indices, row, folder_index)
                    break
                except EndpointConnectionError:
                    tools.log(f"UPLOAD TO S3 - FAILED DUE TO INTERNET CONNECTION  - {folder_name}", logging.WARNING)
                    break
                except S3UploadFailedError:
                    tools.log(f"UPLOAD TO S3 - FAILED DUE TO S3 RELATED PROBLEM  - {folder_name}", logging.WARNING)
                    break
                except Exception:
                    tools.log(f"UPLOAD TO S3 - FAILED DUE TO AN ERROR - {folder_name}", logging.ERROR, exc_info=True)
                    break

            t_delta = time.time() - t0
            timeout_after = timeout_before - t_delta
            if timeout_after <= 0:
                tools.log(
                    f"NEGATIVE TIMEOUT IN upload_analyzed. BEFORE: {timeout_before} AFTER {timeout_after}",
                    log_level=logging.WARNING
                )
            timeout_after = max(10, timeout_after)
            return _customer_code, _plot_code, _scan_date, _uploaded_indices, _uploaded_extensions, _failed_indices, \
                timeout_after

        def send_request(timeout_before, _customer_code, _plot_code, _scan_date, _uploaded_indices,
                         _uploaded_extensions, _failed_indices):

            def add_row_to_dict(d, rows_to_indices, status):
                for row_name, indices in rows_to_indices.items():
                    row_number = row_name.split('_')[1]
                    for index in indices:
                        d["customer_code"].append(_customer_code)
                        d["plot_code"].append(_plot_code)
                        d["scan_date"].append(_scan_date)
                        d["row"].append(row_number)
                        d["folder_index"].append(index)
                        d["status"].append(status)

            t0 = time.time()
            if _uploaded_indices:
                headers = {"Content-Type": "application/json; charset=utf-8", 'Accept': 'text/plain'}
                request_data = {
                    "customer_code": _customer_code,
                    "plot_code": _plot_code,
                    "scan_date": _scan_date,
                    "indices": _uploaded_indices,
                    "file_types": _uploaded_extensions,
                    "output dir": os.path.join(_customer_code, _plot_code, _scan_date),
                    "output types": ['FSI']
                }

                tools.log(f"REQUEST SENT - {_plot_code}: {_uploaded_indices}")
                response = requests.post(data_conf.service_endpoint, json=request_data, headers=headers,
                                         timeout=timeout)
                _response_ok = response.ok
                if _response_ok:
                    tools.log(f"REQUEST SUCCESS - {_plot_code}: {_uploaded_indices}")
                    _uploaded_dict = {
                        "customer_code": [], "plot_code": [], "scan_date": [], "row": [], "folder_index": [],
                        "status": []
                    }
                    add_row_to_dict(_uploaded_dict, _uploaded_indices, consts.success)
                    is_first = not os.path.exists(data_conf.uploaded_path)
                    _uploaded_df = pd.DataFrame(_uploaded_dict)
                    _uploaded_df.to_csv(data_conf.uploaded_path, mode='a+', index=False,
                                        header=is_first)
            else:
                _response_ok = False

            _failed_dict = {
                "customer_code": [], "plot_code": [], "scan_date": [], "row": [], "folder_index": [], "status": []
            }
            add_row_to_dict(_failed_dict, _failed_indices, consts.failed)
            is_first = not os.path.exists(data_conf.uploaded_path)
            _failed_df = pd.DataFrame(_failed_dict)
            _failed_df.to_csv(data_conf.uploaded_path, mode='a+', index=False, header=is_first)

            t_delta = time.time() - t0
            timeout_after = timeout_before - t_delta
            if timeout_after <= 0:
                tools.log(
                    f"NEGATIVE TIMEOUT IN send_request. BEFORE: {timeout_before} AFTER {timeout_after}",
                    log_level=logging.WARNING
                )
            timeout_after = max(10, timeout_after)
            return timeout_after, _response_ok

        analyzed_csv_df = None
        future_timeout = t_scan_start + scan_timeout
        while time.time() < future_timeout:
            try:
                analyzed_csv_df = pd.read_csv(data_conf.analyzed_path, dtype=str)
                break
            except PermissionError:
                time.sleep(5)
            except FileNotFoundError:
                return

        if analyzed_csv_df is None:
            return

        analyzed_csv_df = analyzed_csv_df[analyzed_csv_df["status"] == consts.success]

        uploaded_csv_df = None
        try:
            uploaded_csv_df = pd.read_csv(data_conf.uploaded_path, dtype=str)
        except FileNotFoundError:
            pass

        t_delta = time.time() - t_scan_start
        if uploaded_csv_df is not None:
            analyzed_not_uploaded = pd.merge(analyzed_csv_df, uploaded_csv_df, how='left', indicator=True,
                                             on=["customer_code", "plot_code", "scan_date", "row", "folder_index"])
            not_uploaded = analyzed_not_uploaded['_merge'] == 'left_only'
            analyzed_not_uploaded = analyzed_not_uploaded.loc[not_uploaded, analyzed_not_uploaded.columns != '_merge']
        else:
            analyzed_not_uploaded = analyzed_csv_df

        analyzed_groups = analyzed_not_uploaded.groupby(["customer_code", "plot_code", "scan_date"])
        timeout = scan_timeout - t_delta
        if timeout <= 0:
            tools.log(
                f"NEGATIVE TIMEOUT IN UPLOAD. BEFORE: {scan_timeout} AFTER {timeout}",
                log_level=logging.WARNING
            )
        timeout = max(10, timeout)

        for _, analyzed_gr in analyzed_groups:
            customer_code, plot_code, scan_date, uploaded_indices, uploaded_extensions, failed_indices, \
                timeout = upload_analyzed(timeout, analyzed_gr)
            try:
                timeout, response_ok = send_request(timeout, customer_code, plot_code, scan_date,
                                                    uploaded_indices, uploaded_extensions, failed_indices)
            except:
                tools.log("", logging.ERROR, exc_info=True)
                break

    @staticmethod
    def network_monitor():
        def bytes_to_human(_b):
            suffixes = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
            index = 0
            while _b >= 1024 and index < len(suffixes) - 1:
                _b /= 1024.0
                index += 1
            return f"{_b:.2f} {suffixes[index]}"

        def init_network_dict():
            return {
                    consts.NIC: [],
                    consts.bytes_sent: [],
                    consts.bytes_recv: [],
                    consts.network_sample_timestamp: []
                }

        network_dict = init_network_dict()
        today = datetime.now().strftime('%d%m%y')
        network_log_filename = f"{consts.network_log}_{today}.{consts.log_extension}"
        network_log_path = os.path.join(data_conf.output_path, conf.customer_code, network_log_filename)

        while not DataManager.shutdown_event.is_set():
            net_stats = psutil.net_io_counters(pernic=True)

            # Iterate over network interfaces
            for interface, stats in net_stats.items():
                network_sample_timestamp = datetime.now().strftime(data_conf.timestamp_format)
                if stats.bytes_sent > 0 and stats.bytes_recv > 0:
                    network_dict[consts.NIC].append(interface)
                    network_dict[consts.bytes_sent].append(bytes_to_human(stats.bytes_sent))
                    network_dict[consts.bytes_recv].append(bytes_to_human(stats.bytes_recv))
                    network_dict[consts.network_sample_timestamp].append(network_sample_timestamp)

            if len(network_dict[consts.NIC]) % 20 == 0 and len(network_dict[consts.NIC]) > 0:
                try:
                    is_first = not os.path.exists(network_log_path)
                    pd.DataFrame(network_dict).to_csv(network_log_path, mode='a+', header=is_first, index=False)
                    network_dict = init_network_dict()
                except:
                    traceback.print_exc()

            time.sleep(consts.network_traffic_sleep_duration)

    @staticmethod
    def jtop_logger():
        try:
            today = datetime.now().strftime(data_conf.date_format)
            jtop_log_filename = f"{consts.jtop_log}_{today}.{consts.log_extension}"
            jtop_log_path = os.path.join(data_conf.output_path, conf.customer_code, jtop_log_filename)

            with jtop() as jetson:
                # Create an empty DataFrame with the same column names as the statistics
                stats = jetson.stats
                jtop_df = pd.DataFrame(columns=stats.keys())

                while jetson.ok():
                    jtop_df = jtop_df.append(jetson.stats, ignore_index=True)
                    if len(jtop_df) % 20 == 0:
                        is_first = not os.path.exists(jtop_log_path)
                        jtop_df.to_csv(jtop_log_path, mode='a+', header=is_first, index=False)
                        jtop_df = pd.DataFrame(columns=stats.keys())

        except JtopException as e:
            print(e)

