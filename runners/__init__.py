from .BasicRunner import BasicRunner
# from .MegaCRNRunner import MegaCRNRunner
# from .GCRNRunner import GCRNRunner


def runner_select(name):
    name = name.upper()

    if name == "BASIC":
        return BasicRunner
    else:
        raise NotImplementedError
