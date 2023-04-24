import enum
import time
import os
import signal
import threading
import logging
from builtins import staticmethod
from multiprocessing import Process, Pipe, current_process

from typing import List


class DataError(Exception):
    """ raised when there is no data to receive """

    def __init__(self):
        super().__init__("No Data")


class ModulesEnum(enum.Enum):
    GPS = "GPS"
    GUI = "GUI"
    DataManager = "DATA MANAGER"
    Analysis = "ANALYSIS"

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        return self.value == other.value


class ModuleTransferAction(enum.Enum):
    NAV = "NAV"
    IMU = "IMU"
    FRUITS_DATA = "FRUITS_DATA"
    BLOCK_SWITCH = "BLOCK_SWITCH"
    START_ACQUISITION = "START_ACQUISITION"
    STOP_ACQUISITION = "STOP_ACQUISITION"
    VIEW_START = "VIEW_START"
    VIEW_STOP = "VIEW_STOP"
    GUI_SET_DEVICE_STATE = "GUI_SET_DEVICE_STATE"

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        return self.value == other.value

    def __str__(self):
        return self.value


class ModuleManager:
    def __init__(self):
        self._process = None
        self.pid = -1
        self.module_name = None
        self.sender, self.receiver = Pipe()

    def set_process(self, target, main_pid, module_name):
        self.module_name = module_name
        args = (self.sender, self.receiver, main_pid, module_name)
        self._process = Process(target=target, args=args, daemon=True, name=module_name.value)

    def retrieve_transferred_data(self):
        if self.receiver.poll():
            return self.receiver.recv()
        raise DataError

    def receive_transferred_data(self, data, sender_module):
        self.sender.send((data, sender_module))
        os.kill(self.pid, signal.SIGUSR1)

    def start(self):
        self._process.start()
        self.pid = self._process.pid
        logging.info(f"{self.module_name} PID: {self.pid}")

    def join(self):
        self._process.join()

    def shutdown(self):
        os.kill(self.pid, signal.SIGTERM)


class Module:
    """ An abstraction class for all modules """
    main_pid, sender, receiver, module_name = -1, None, None, None
    shutdown_event = threading.Event()
    shutdown_done_event = threading.Event()

    @staticmethod
    def init_module(sender, receiver, main_pid, module_name):
        Module.sender = sender
        Module.receiver = receiver
        Module.main_pid = main_pid
        Module.module_name = module_name

    @staticmethod
    def set_signals(shutdown_func, receive_data_func):
        signal.signal(signal.SIGTERM, shutdown_func)
        signal.signal(signal.SIGUSR1, receive_data_func)

    @staticmethod
    def send_data(action, data, *receivers: ModulesEnum):
        data = {
            "action": action,
            "data": data
        }
        Module.sender.send((data, receivers))
        os.kill(Module.main_pid, signal.SIGUSR1)

    @staticmethod
    def receive_data(sig, frame):
        """ every module has to implement that on his own """
        pass

    @staticmethod
    def shutdown(sig, frame):
        Module.shutdown_event.set()
        while not Module.shutdown_done_event.is_set():
            time.sleep(5)
