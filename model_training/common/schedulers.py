import math
from typing import Dict

import torch.optim as optim
from torch.optim import lr_scheduler

from torch.optim.optimizer import Optimizer

available_schedulers = {
    "plateau": ["mode", "patience", "factor", "min_lr"],
    "MultiStepLR": ["milestones", "gamma"],
}


def get_scheduler(config: Dict, optimizer: Optimizer):
    """ Creates scheduler for a given optimizer.
    :param config:
    :param optimizer:
    :return: torch.optim.lr_scheduler._LRScheduler: optimizer scheduler
    """
    scheduler_config = config["scheduler"]

    if scheduler_config["name"] == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=scheduler_config["mode"],
            patience=scheduler_config["patience"],
            factor=scheduler_config["factor"],
            min_lr=scheduler_config["min_lr"],
        )
    elif scheduler_config["name"] == "MultiStepLR":
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, scheduler_config["milestones"], gamma=scheduler_config["gamma"]
        )
    else:
        raise ValueError("Scheduler [%s] not recognized." % scheduler_config["name"])

    return scheduler


class WarmRestart(lr_scheduler.CosineAnnealingLR):
    """This class implements Stochastic Gradient Descent with Warm Restarts(SGDR):
    https://arxiv.org/abs/1608.03983.

    Set the learning rate of each parameter group using a cosine annealing schedule,
    When last_epoch=-1, sets initial lr as lr.
    This can't support scheduler.step(epoch). please keep epoch=None.
    """

    def __init__(self, optimizer, T_max=30, T_mult=1, eta_min=0, last_epoch=-1):
        """implements SGDR

        Parameters:
        ----------
        T_max : int
            Maximum number of epochs.
        T_mult : int
            Multiplicative factor of T_max.
        eta_min : int
            Minimum learning rate. Default: 0.
        last_epoch : int
            The index of last epoch. Default: -1.
        """
        self.T_mult = T_mult
        super().__init__(optimizer, T_max, eta_min, last_epoch)

    def get_lr(self):
        if self.last_epoch == self.T_max:
            self.last_epoch = 0
            self.T_max *= self.T_mult
        return [
            self.eta_min
            + (base_lr - self.eta_min)
            * (1 + math.cos(math.pi * self.last_epoch / self.T_max))
            / 2
            for base_lr in self.base_lrs
        ]


class LinearDecay(lr_scheduler._LRScheduler):
    """This class implements LinearDecay

    """

    def __init__(self, optimizer, num_epochs, start_epoch=0, min_lr=0, last_epoch=-1):
        """implements LinearDecay

        Parameters:
        ----------

        """
        self.num_epochs = num_epochs
        self.start_epoch = start_epoch
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.start_epoch:
            return self.base_lrs
        return [
            base_lr
            - ((base_lr - self.min_lr) / self.num_epochs)
            * (self.last_epoch - self.start_epoch)
            for base_lr in self.base_lrs
        ]
