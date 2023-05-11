from application.utils.module_wrapper import ModulesEnum, Module, ModuleTransferAction
import signal

class AlternativeFlow(Module):

    @staticmethod
    def init_module(sender, receiver, main_pid, module_name):
        super(AlternativeFlow, AlternativeFlow).init_module(sender, receiver, main_pid, module_name)
        signal.signal(signal.SIGTERM, AlternativeFlow.shutdown)
        signal.signal(signal.SIGUSR1, AlternativeFlow.receive_data)

    @staticmethod
    def shutdown(sig, frame):
        pass

    @staticmethod
    def receive_data(sig, frame):
        pass
