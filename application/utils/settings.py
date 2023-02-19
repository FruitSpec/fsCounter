from omegaconf import OmegaConf
import os
import logging
from datetime import datetime

today = datetime.utcnow().strftime('%d-%m-%Y')

data_conf = OmegaConf.load(os.path.abspath("application/DataManager/Data_config.yaml"))
GPS_conf = OmegaConf.load(os.path.abspath("application/GPS/GPS_config.yaml"))
GUI_conf = OmegaConf.load(os.path.abspath("application/GUI/GUI_config.yaml"))
conf = OmegaConf.load(os.path.abspath("application/utils/config.yaml"))

log_path = os.path.abspath(os.path.expanduser(conf["logs path"]))
conf["log name"] = os.path.join(log_path, f"{conf['counter number']}_{conf['log name']}_{today}.log")
if not os.path.exists(log_path):
    os.makedirs(log_path)

logFormatter = logging.Formatter("%(levelname)s %(asctime)s %(processName)s %(message)s")
fileHandler = logging.FileHandler(conf["log name"])
fileHandler.setFormatter(logFormatter)
rootLogger = logging.getLogger()
rootLogger.addHandler(fileHandler)
rootLogger.setLevel(logging.INFO)
