import os
import logging
import time
import traceback
from datetime import datetime
import requests
import boto3
import speedtest
from boto3.exceptions import S3UploadFailedError
from botocore.config import Config
from botocore.exceptions import EndpointConnectionError
from application.utils.settings import conf, data_conf, GPS_conf, analysis_conf, GUI_conf


def s3_path_join(*args):
    return "/".join(args)


def index_from_fruits(filename, with_prefix=True, as_int=False):
    start = -8 if with_prefix else -7
    end = -4
    if not is_csv(filename):
        start -= 4
        end -= 4
    return int(filename[start:end]) if as_int else filename[start:end]


def index_from_svo(filename, as_int=False):
    svo_index = filename.split('_')[1].split('.')[0]
    return int(svo_index) if as_int else svo_index


def is_csv(filename):
    return filename[-3:] == "csv"


def is_svo(filename):
    return filename[-3:] == "svo"


def get_nav_path():
    today = datetime.now().strftime("%d%m%y")
    nav_dir = os.path.join(data_conf["output path"], conf["customer code"])
    if not os.path.exists(nav_dir):
        os.makedirs(nav_dir)
    return os.path.join(nav_dir, f'{today}.nav')


def get_imu_path():
    today = datetime.now().strftime("%d%m%y")
    return os.path.join(data_conf["output path"], conf["customer code"], f'{today}.imu')


def get_fruits_path(plot, row, index=-1, write_csv=True, get_dir=False):
    ext = "csv" if is_csv else "feather"
    today = datetime.now().strftime("%d%m%y")
    row = f"row_{row}"
    plot_dir = os.path.join(data_conf["output path"], conf["customer code"], plot, today, row)
    if get_dir:
        return plot_dir
    index_str = f"{data_conf['file prefix']}{str(index).zfill(3)}"
    filename = f"fruits_{index_str}.{ext}"
    return os.path.join(plot_dir, filename)


def upload_to_s3(customer_code, plot_code, scan_date, indices, timeout):
    def get_file_size(f):
        return os.path.getsize(f) / 1024

    def timed_call(timeout_before, func, *args):
        t0 = time.time()
        if args:
            ret = func(*args)
        else:
            ret = func()
        t1 = time.time()
        timeout_after = timeout_before - (t1 - t0)
        if ret:
            return ret, timeout_after
        else:
            return timeout_after

    valid_indices = []

    # iterate through the indices and separate the files by size
    file_suffix = "feather" if data_conf["use feather"] else "csv"
    try:
        # get the upload speed (bit/s) and remaining time
        upload_in_bps, timeout = timed_call(timeout, speedtest.Speedtest().upload)

        # convert into KB/s (kilobytes - not kilo bits)
        upload_in_kbps = upload_in_bps / (1024 * 8)
        if upload_in_kbps < 10:
            # this will cause the scan to act as if there was no internet
            raise speedtest.SpeedtestException
    except speedtest.SpeedtestException:
        logging.info("DATA MANAGER - NO INTERNET CONNECTION")
        return False, []

    logging.info(f"DATA MANAGER - INTERNET UPLOAD SPEED - {upload_in_kbps} KB/s")
    s3_client = boto3.client("s3", config=Config(retries={"total_max_attempts": 1}))

    current_path = os.path.join(customer_code, plot_code, scan_date)

    logging.info(f"DATA MANAGER - UPLOAD TO S3 (TRYING) - {current_path}")
    logging.info(f"DATA MANAGER - UPLOAD TO S3 (TRYING) - UPLOADING INDICES {indices}")
    for index in indices:
        filename = f"fruits_{index}.{file_suffix}"
        file_path = os.path.join(data_conf["output path"], customer_code, plot_code, scan_date, filename)
        f_size = get_file_size(file_path)
        if timeout <= 1:
            logging.info(f"DATA MANAGER - UPLOAD TO S3 - TIMEOUT STATUS - {timeout} - STOPPING")
            break

        filename = f"fruits_{index}.{file_suffix}"
        fruits_path = os.path.join(data_conf["output path"], customer_code, plot_code, scan_date, filename)
        aws_path = os.path.join(customer_code, plot_code, scan_date, filename)

        # f_size is the size of the file in KB.
        # upload_in_kbps * timeout gives us the approximated amount of data that can be uploaded within given timeout/
        # if, by our approximation, the file won't upload in time, we don't even try.
        if f_size >= upload_in_kbps * timeout:
            continue
        try:
            # Attempt to upload the file and update the remaining time
            timeout = timed_call(timeout, s3_client.upload_file, fruits_path, data_conf["upload bucket name"],
                                 aws_path)
            logging.info(f"DATA MANAGER - UPLOAD TO S3 - {current_path}/{filename} - SUCCESS, TIMEOUT STATUS - {timeout}")
            valid_indices.append(index)
        except TimeoutError:
            break
        except FileNotFoundError:
            logging.warning(f"DATA MANAGER - UPLOAD TO S3 - MISSING INDEX - {index}")
        except EndpointConnectionError:
            logging.warning(f"DATA MANAGER - UPLOAD TO S3 FAILED - INTERNET CONNECTION - {current_path}")
            return False, []
        except S3UploadFailedError:
            logging.warning(f"DATA MANAGER - UPLOAD TO S3 FAILED - S3 RELATED PROBLEM - {current_path}")
            return False, []
        except Exception:
            logging.error(f"DATA MANAGER - UPLOAD TO S3 FAILED - UNKNOWN ERROR (see traceback) - {current_path}")
            traceback.print_exc()
            return False, []

    logging.info(f"UPLOAD TO S3 - SUCCESS - {current_path} - INDICES - {valid_indices}")
    return True, valid_indices


def send_request_to_server(customer_code, plot_code, scan_date, indices):
    file_type = "FEATHER" if data_conf["use feather"] else "CSV"
    fruits_data = {
        "bucket": conf["upload bucket name"],
        "prefix": conf["upload prefix"],
        "customer_code": customer_code,
        "plot_code": plot_code,
        "date": scan_date,
        "indices": indices,
        # "project type": settings.project_type,
        # "season": settings.season,
        "file type": file_type,
        "counter number": conf["counter number"]
    }

    headers = {"Content-Type": "application/json; charset=utf-8", "Accept": "text/plain"}
    return requests.post(data_conf["service endpoint"], json=fruits_data, headers=headers,
                         timeout=data_conf["request timeout"])
