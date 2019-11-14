import torch.optim as optim
from typing import Dict

from torch.optim.optimizer import Optimizer

available_optimizers = ["adam", "sgd", "adadelta"]


def get_optimizer(config: Dict, params) -> Optimizer:
    """
    Creates model optimizer from Trainer config.
    :param config:
    :param params: list of model parameters to be trained
    :return: torch.optim.optimizer.Optimizer: model optimizer
    """
    name = config["optimizer"]["name"]
    lr = config["optimizer"]["lr"]
    momentum = config["optimizer"].get("momentum", 0)
    weight_decay = config["optimizer"].get("weight_decay", 0)

    optimizers = {
        "adam": lambda: optim.Adam(params, lr=lr),
        "sgd": lambda: optim.SGD(
            params, lr=lr, momentum=momentum, weight_decay=weight_decay
        ),
        "adadelta": lambda: optim.Adadelta(params, lr=lr),
    }

    try:
        return optimizers[name]()
    except KeyError:
        raise KeyError("Optimizer [{}] not recognized.".format(name))
