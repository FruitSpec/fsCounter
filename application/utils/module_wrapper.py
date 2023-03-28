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

    def __hash__(self):
        return hash(self.value)

    def __eq__(self, other):
        return self.value == other.value


class ModuleManager:
    def __init__(self):
        self._process = None
        self.pid = -1
        self.sender, self.receiver = Pipe()

    def set_process(self, target, main_pid, module_name):
        args = (self.sender, self.receiver, main_pid, module_name)
        self._process = Process(target=target, args=args, daemon=True)

    def retrieve_transferred_data(self):
        if self.receiver.poll():
            return self.receiver.recv()
        raise DataError

    def receive_transferred_data(self, data, sender_module):
        print("receive_transferred_data - signaling to pid ", self.pid)
        self.sender.send((data, sender_module))
        print("Here 1")
        os.kill(self.pid, signal.SIGUSR1)
        print("Here 2")

    def start(self):
        self._process.start()
        self.pid = self._process.pid
        print(self.pid)

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
    def send_data(data, *receivers: ModulesEnum):
        Module.sender.send((data, receivers))
        os.kill(Module.main_pid, signal.SIGUSR1)

    @staticmethod
    def receive_data(sig, frame):
        print("Module receive_data:")

        """ every module has to implement that on his own """
        pass

    @staticmethod
    def shutdown(sig, frame):
        Module.shutdown_event.set()
        while not Module.shutdown_done_event.is_set():
            time.sleep(5)

