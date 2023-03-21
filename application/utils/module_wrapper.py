import enum
import time
import os
import signal
import threading
from builtins import staticmethod
from multiprocessing import Process, Pipe

from typing import List


class DataError(Exception):
    """ raised when there is no data to receive """
    def __init__(self):
        super().__init__("No Data")


class ModulesEnum(enum.Enum):
    GPS = "GPS"
    GUI = "GUI"
    DataManager = "Data Manager"
    Analysis = "Analysis"


class ModuleManager:
    def __init__(self):
        self._process = None
        self.pid = -1
        self.sender, self.receiver = Pipe()

    def set_process(self, *args, target, main_pid):
        args = (self.sender, self.receiver, main_pid) + args
        self._process = Process(target=target, args=args, daemon=True)
        self.pid = self._process.pid

    def retrieve_transferred_data(self):
        if self.receiver.poll():
            return self.receiver.recv()
        raise DataError

    def receive_transferred_data(self, data, sender_module):
        self.sender.send((data, sender_module))
        os.kill(self.pid, signal.SIGUSR1)

    def start(self):
        self._process.start()

    def join(self):
        self._process.join()

    def shutdown(self):
        os.kill(self.pid, signal.SIGTERM)


class Module:
    """ An abstraction class for all modules """
    main_pid, sender, receiver = -1, None, None
    shutdown_event = threading.Event()
    shutdown_done_event = threading.Event()

    @staticmethod
    def init_module(sender, receiver, main_pid):
        Module.sender = sender
        Module.receiver = receiver
        Module.main_pid = main_pid

    @staticmethod
    def set_signals(shutdown_func, receive_data_func):
        signal.signal(signal.SIGTERM, shutdown_func)
        signal.signal(signal.SIGUSR1, receive_data_func)

    @staticmethod
    def send_data(data, *receivers: ModulesEnum):
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

