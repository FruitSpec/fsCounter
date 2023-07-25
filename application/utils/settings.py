from omegaconf import OmegaConf
import os
import logging
from datetime import datetime

today = datetime.now().strftime('%d-%m-%Y')

conf = OmegaConf.load(os.path.abspath("application/utils/config.yaml"))
crop = conf.crop

consts = OmegaConf.load(os.path.abspath("application/utils/consts.yaml"))
analysis_conf = OmegaConf.load(os.path.abspath("application/Analysis/analysis_config.yaml"))[crop]
data_conf = OmegaConf.load(os.path.abspath("application/DataManager/data_config.yaml"))
GPS_conf = OmegaConf.load(os.path.abspath("application/GPS/GPS_config.yaml"))
GUI_conf = OmegaConf.load(os.path.abspath("application/GUI/GUI_config.yaml"))
pipeline_conf = OmegaConf.load(os.path.abspath("application/Analysis/pipeline_config.yaml"))
runtime_args = OmegaConf.load(os.path.abspath("application/Analysis/runtime_config.yaml"))

log_path = os.path.abspath(os.path.expanduser(conf.logs_path))
conf.log_name = os.path.join(log_path, f"{conf.counter_number}_{conf.log_name}_{today}.log")
if not os.path.exists(log_path):
    os.makedirs(log_path)


def set_logger():
    log_formatter = logging.Formatter("%(asctime)s | %(levelname)s/%(processName)s | %(message)s", "%H:%M:%S")
    file_handler = logging.FileHandler(conf.log_name)
    file_handler.setFormatter(log_formatter)
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    root_logger.setLevel(logging.INFO)
