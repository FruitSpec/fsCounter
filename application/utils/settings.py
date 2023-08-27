from omegaconf import OmegaConf
import os
import logging
from datetime import datetime

conf = OmegaConf.load(os.path.abspath("application/utils/config.yaml"))
crop = conf.crop

consts = OmegaConf.load(os.path.abspath("application/utils/consts.yaml"))
analysis_conf = OmegaConf.load(os.path.abspath("application/Analysis/analysis_config.yaml"))[crop]
data_conf = OmegaConf.load(os.path.abspath("application/DataManager/data_config.yaml"))
GPS_conf = OmegaConf.load(os.path.abspath("application/GPS/GPS_config.yaml"))
GUI_conf = OmegaConf.load(os.path.abspath("application/GUI/GUI_config.yaml"))
pipeline_conf = OmegaConf.load(os.path.abspath("application/Analysis/pipeline_config.yaml"))
runtime_args = OmegaConf.load(os.path.abspath("application/Analysis/runtime_config.yaml"))


def set_logger():
    today = datetime.now().strftime('%d%m%y')
    log_basename = f"{conf.counter_number}_{consts.log_name}_{today}.{consts.log_extension}"
    log_path = os.path.join(consts.log_dir, log_basename)
    if not os.path.exists(consts.log_dir):
        os.makedirs(consts.log_dir)

    log_formatter = logging.Formatter("%(asctime)s | %(levelname)s/%(processName)s | %(message)s", "%H:%M:%S")
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(log_formatter)
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)
    root_logger.setLevel(logging.INFO)
