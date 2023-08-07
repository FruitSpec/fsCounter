import enum
import queue
import time
import os
import signal
import threading
import logging
import traceback
from builtins import staticmethod
from multiprocessing import Process, Queue


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
    Main = "MAIN"

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        try:
            return self.value == other.value
        except AttributeError:
            print(other)
            traceback.print_exc()


class ModuleTransferAction(enum.Enum):
    MONITOR = "MONITOR"
    RESTART_APP = "RESTART_APP"
    START_GPS = "START_GPS"
    NAV = "NAV"
    IMU = "IMU"
    ASK_FOR_NAV = "ASK_FOR_NAV"
    JAIZED_TIMESTAMPS = "JAIZED_TIMESTAMPS"
    JAIZED_TIMESTAMPS_AND_STOP = "JAIZED_TIMESTAMPS_AND_STOP"
    ANALYSIS_ONGOING = "ANALYSIS_ONGOING"
    ANALYSIS_DONE = "ANALYSIS_DONE"
    ANALYZED_DATA = "ANALYZED_DATA"
    FRUITS_DATA = "FRUITS_DATA"
    ENTER_PLOT = "ENTER_PLOT"
    EXIT_PLOT = "EXIT_PLOT"
    START_ACQUISITION = "START_ACQUISITION"
    STOP_ACQUISITION = "STOP_ACQUISITION"
    ACQUISITION_CRASH = "ACQUISITION_CRASH"
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
    def __init__(self, main_pid, communication_queue):
        self.notify_on_death = None
        self.death_action = None
        self._process = None
        self.pid = -1
        self.target = None
        self.main_pid = main_pid
        self.module_name = None
        self.daemon = True
        self.communication_queue = communication_queue
        self.in_qu = Queue()
        self.out_qu = Queue()

    def __str__(self):
        return self.module_name

    def set_process(self, target, module_name, notify_on_death=None, death_action=None, daemon=True):
        self.target = target
        self.module_name = module_name
        self.daemon = daemon
        self.notify_on_death = notify_on_death
        self.death_action = death_action
        args = (self.in_qu, self.out_qu, self.main_pid, module_name, self.communication_queue, notify_on_death,
                death_action)
        self._process = Process(target=target, args=args, daemon=daemon, name=module_name.value)

    def respawn(self):
        args = (self.in_qu, self.out_qu, self.main_pid, self.module_name, self.communication_queue,
                self.notify_on_death, self.death_action)
        self._process = Process(target=self.target, args=args, daemon=self.daemon, name=self.module_name.value)
        self.start(is_respawn=True)

    def is_alive(self):
        return self._process.is_alive()

    def retrieve_transferred_data(self):
        try:
            return self.out_qu.get(timeout=1)
        except queue.Empty:
            raise DataError

    def receive_transferred_data(self, data, sender_module):
        try:
            self.in_qu.put((data, sender_module), timeout=1)
        except queue.Empty:
            raise DataError

    def start(self, is_respawn=False):
        self._process.start()
        self.pid = self._process.pid
        if is_respawn:
            print(f"{self.module_name} PROCESS RESPAWNED - NEW PID: {self.pid}")
            logging.info(f"{self.module_name} PROCESS RESPAWNED - NEW PID: {self.pid}")
        else:
            print(f"{self.module_name} PID: {self.pid}")
            logging.info(f"{self.module_name} PID: {self.pid}")
        return self.pid

    def join(self):
        self._process.join()

    def terminate(self):
        os.kill(self.pid, signal.SIGTERM)


class Module:
    """ An abstraction class for all modules """
    # main_pid, sender, receiver, module_name = -1, None, None, None
    main_pid, in_qu, out_qu, module_name = -1, None, None, None
    communication_queue = None
    shutdown_event = threading.Event()
    shutdown_done_event = threading.Event()

    @staticmethod
    def init_module(in_qu, out_qu, main_pid, module_name, communication_queue, notify_on_death, death_action):
        Module.in_qu = in_qu
        Module.out_qu = out_qu
        Module.main_pid = main_pid
        Module.module_name = module_name
        Module.communication_queue = communication_queue
        Module.notify_on_death = notify_on_death
        Module.death_action = death_action

    @staticmethod
    def set_signals(shutdown_func):
        signal.signal(signal.SIGTERM, shutdown_func)

    @staticmethod
    def send_data(action, data, receiver, require_ack=False):
        data = {
            "action": action,
            "data": data
        }
        logging.info(f"SENDING DATA")
        try:
            Module.communication_queue.put(Module.module_name, timeout=1)
            Module.out_qu.put((data, receiver))
        except queue.Full:
            logging.warning("COMMUNICATION QUEUE IS FULL!")
            return
        # os.kill(Module.main_pid, signal.SIGUSR1)

    @staticmethod
    def receive_data():
        """ every module has to implement that on his own """
        pass

    @staticmethod
    def shutdown(sig, frame):
        print(f"SHUTDOWN RECEIVED IN PROCESS {Module.module_name}")
        logging.warning(f"SHUTDOWN RECEIVED IN PROCESS {Module.module_name}")
        Module.shutdown_event.set()
        while not Module.shutdown_done_event.is_set():
            time.sleep(5)
