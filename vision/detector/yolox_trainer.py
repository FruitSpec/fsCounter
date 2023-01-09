import os
import sys

from vision.misc.help_func import get_repo_dir



repo_dir = get_repo_dir()
sys.path.append(os.path.join(repo_dir, 'vision', 'detector', 'yolo_x'))

from yolo_x.yolox.core import trainer
from yolo_x.yolox.exp import Exp, get_exp


trainer