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
from application.utils import tools


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
            tools.log(str(other), log_level=logging.ERROR, exc_info=True)


class ModuleTransferAction(enum.Enum):
    SET_LOGGER = "SET_LOGGER"
    MONITOR = "MONITOR"
    REBOOT = "REBOOT"
    RESTART_APP = "RESTART_APP"
    CONNECT_CAMERAS = "CONNECT_CAMERAS"
    START_GPS = "START_GPS"
    NAV = "NAV"
    ASK_FOR_NAV = "ASK_FOR_NAV"
    JAIZED_TIMESTAMPS = "JAIZED_TIMESTAMPS"
    JAIZED_TIMESTAMPS_AND_STOP = "JAIZED_TIMESTAMPS_AND_STOP"
    ANALYSIS_ONGOING = "ANALYSIS_ONGOING"
    ANALYSIS_DONE = "ANALYSIS_DONE"
    ANALYZED_DATA = "ANALYZED_DATA"
    FRUITS_DATA = "FRUITS_DATA"
    ENTER_PLOT = "ENTER_PLOT"
    EXIT_PLOT = "EXIT_PLOT"
    START_RECORDING = "START_RECORDING"
    STOP_RECORDING = "STOP_RECORDING"
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

    def set_logger(self):
        return self._process.is_alive()

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
            tools.log(f"{self.module_name} PROCESS RESPAWNED - NEW PID: {self.pid}")
        else:
            tools.log(f"{self.module_name} PID: {self.pid}")
        return self.pid

    def join(self):
        self._process.join()

    def terminate(self):
        try:
            os.kill(self.pid, signal.SIGTERM)
        except:
            tools.log(f"COULD NOT KILL MODULE {self.module_name}")

    def force_kill(self):
        try:
            os.kill(self.pid, signal.SIGKILL)
            tools.log(f"FORCE-KILL MODULE {self.module_name}")
        except:
            pass


class Module:
    """ An abstraction class for all modules """
    # main_pid, sender, receiver, module_name = -1, None, None, None
    main_pid, in_qu, out_qu, module_name = -1, None, None, None
    communication_queue = None
    shutdown_event = threading.Event()

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
    def send_data(action, data, receiver, log_option=tools.LogOptions.LOG):
        data = {
            "action": action,
            "data": data,
            "log_option": log_option
        }
        tools.log(f"SENDING DATA", log_option=log_option)
        try:
            try:
                Module.communication_queue.put(Module.module_name, timeout=1)
            except queue.Full:
                tools.log("COMMUNICATION QUEUE IS FULL!", logging.WARNING)
                return
            try:
                Module.out_qu.put((data, receiver), timeout=1)
            except queue.Full:
                tools.log("OUT QUEUE IS FULL!", logging.WARNING)
        except:
            tools.log("UNDETECTED PROBLEM IN send_data", log_level=logging.ERROR, exc_info=True)

    @staticmethod
    def receive_data():
        """ every module has to implement that on his own """
        pass

    @staticmethod
    def shutdown(sig, frame):
        tools.log(f"SHUTDOWN RECEIVED IN PROCESS {Module.module_name}", logging.WARNING)
        Module.shutdown_event.set()
        time.sleep(3)
