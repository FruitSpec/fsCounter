import enum
import queue
import time
import os
import signal
import threading
import logging
import traceback
from builtins import staticmethod
from multiprocessing import Process, Pipe, Queue

from typing import List


class DataError(Exception):
    """ raised when there is no data to receive """

    def __init__(self):
        super().__init__("No Data")


class ModulesEnum(enum.Enum):
    GPS = "GPS"
    GUI = "GUI"
    DataManager = "DATA MANAGER"
    Acquisition = "ACQUISITION"
    Analysis = "ANALYSIS"

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        try:
            return self.value == other.value
        except AttributeError:
            print(other)
            traceback.print_exc()


class ModuleTransferAction(enum.Enum):
    NAV = "NAV"
    IMU = "IMU"
    JAIZED_TIMESTAMPS = "JAIZED_TIMESTAMPS"
    JAIZED_TIMESTAMPS_AND_STOP = "JAIZED_TIMESTAMPS_AND_STOP"
    ANALYZED_DATA = "ANALYZED_DATA"
    FRUITS_DATA = "FRUITS_DATA"
    ENTER_PLOT = "ENTER_PLOT"
    EXIT_PLOT = "EXIT_PLOT"
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
        self.communication_queue = None
        self.in_qu = Queue()
        self.out_qu = Queue()

    def set_process(self, target, main_pid, module_name, communication_queue, daemon=True):
        self.module_name = module_name
        # args = (self.sender, self.receiver, main_pid, module_name)
        args = (self.in_qu, self.out_qu, main_pid, module_name, communication_queue)
        self._process = Process(target=target, args=args, daemon=daemon, name=module_name.value)

    def retrieve_transferred_data(self):
        try:
            return self.out_qu.get_nowait()
        except queue.Empty:
            raise DataError

    def receive_transferred_data(self, data, sender_module):
        # self.sender.send((data, sender_module))
        self.in_qu.put((data, sender_module))
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
    # main_pid, sender, receiver, module_name = -1, None, None, None
    main_pid, in_qu, out_qu, module_name = -1, None, None, None
    communication_queue = None
    shutdown_event = threading.Event()
    shutdown_done_event = threading.Event()

    @staticmethod
    def init_module(in_qu, out_qu, main_pid, module_name, communication_queue):
        Module.in_qu = in_qu
        Module.out_qu = out_qu
        Module.main_pid = main_pid
        Module.module_name = module_name
        Module.communication_queue = communication_queue

    @staticmethod
    def set_signals(shutdown_func, receive_data_func):
        signal.signal(signal.SIGTERM, shutdown_func)
        signal.signal(signal.SIGUSR1, receive_data_func)

    @staticmethod
    def send_data(action, data, receiver):
        data = {
            "action": action,
            "data": data
        }
        Module.out_qu.put((data, receiver))
        Module.communication_queue.put(Module.module_name)
        time.sleep(0.1)
        os.kill(Module.main_pid, signal.SIGUSR1)

    @staticmethod
    def receive_data(sig, frame):
        """ every module has to implement that on his own """
        pass

    @staticmethod
    def shutdown(sig, frame):
        print(f"SHUTDOWN RECEIVED IN PROCESS {Module.module_name}")
        logging.warning(f"SHUTDOWN RECEIVED IN PROCESS {Module.module_name}")
        Module.shutdown_event.set()
        while not Module.shutdown_done_event.is_set():
            time.sleep(5)
