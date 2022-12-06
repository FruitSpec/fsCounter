import signal
import subprocess
from time import sleep
from GPS import location_awareness
from DataManager import uploader
from Analysis import analyzer
from utils.module_wrapper import ModuleManager, DataError, ModulesEnum
from application.utils.settings import conf

global manager


def shutdown():
    manager[ModulesEnum.GPS].shutdown()
    manager[ModulesEnum.DataManager].shutdown()
    manager[ModulesEnum.Analysis].shutdown()


def transfer_data(sig, frame):
    for sender_module in ModulesEnum:
        try:
            data, recv_module = manager[sender_module].get_data()
            manager[recv_module].transfer_data(data, sender_module)
        except DataError:
            continue


def main():
    global manager
    manager = dict()
    if conf["GUI"]:
        subprocess.Popen("npm run dev --prefix ./JAI-Operator-client", shell=True)

    for _, module in enumerate(ModulesEnum):
        manager[module] = ModuleManager()

    manager[ModulesEnum.GPS].set_process(target=location_awareness.GPSSampler.init_module)
    manager[ModulesEnum.DataManager].set_process(target=uploader.init_module)
    manager[ModulesEnum.Analysis].set_process(target=analyzer.init_module)

    manager[ModulesEnum.GPS].start()
    manager[ModulesEnum.DataManager].start()
    manager[ModulesEnum.Analysis].start()

    signal.signal(signal.SIGUSR1, transfer_data)

    manager[ModulesEnum.GPS].join()
    manager[ModulesEnum.DataManager].join()
    manager[ModulesEnum.Analysis].join()


if __name__ == "__main__":
    main()
