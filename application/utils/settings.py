from omegaconf import OmegaConf
import os
import logging
from datetime import datetime

today = datetime.utcnow().strftime('%d-%m-%Y')

data_conf = OmegaConf.load(os.path.abspath("DataManager/Data_config.yaml"))
GPS_conf = OmegaConf.load(os.path.abspath("GPS/GPS_config.yaml"))
GUI_conf = OmegaConf.load(os.path.abspath("GUI/GUI_config.yaml"))
conf = OmegaConf.load(os.path.abspath("utils/config.yaml"))
conf["log name"] = os.path.join(conf["logs path"], f"{conf['log name']}_{today}.log")


# logFormatter = logging.Formatter("%(levelname)s %(asctime)s %(processName)s %(message)s")
# fileHandler = logging.FileHandler("{0}".format(conf["log name"]))
# fileHandler.setFormatter(logFormatter)
# rootLogger = logging.getLogger()
# rootLogger.addHandler(fileHandler)
# rootLogger.setLevel(logging.INFO)
