from collections import defaultdict
from typing import Dict

import numpy as np
from tensorboardX import SummaryWriter


class AbstractMetricCounter:
    def __init__(self, experiment_directory: str):
        self.writer = SummaryWriter(log_dir=experiment_directory)

        self.metrics = defaultdict(list)
        self.best_metric = -1
        self.window_size = 100

    def add_losses(self, loss_dict):
        self.metrics["Loss"].append(loss_dict["loss"])

    def loss_message(self):
        mean_loss_from_window = np.mean(self.metrics["Loss"][-self.window_size :])
        return "Loss={0:.4f}".format(mean_loss_from_window)

    def get_metric(self):
        raise NotImplementedError()

    def clear(self):
        self.metrics = defaultdict(list)

    def write_to_tensorboard(self, epoch_num, validation=False):
        prefix = "Validation" if validation else "Train"

        for k in filter(lambda val: val != "default", self.metrics.keys()):
            self.writer.add_scalar(
                tag=f"{prefix}_{k}",
                scalar_value=np.mean(self.metrics[k]),
                global_step=epoch_num,
            )

        self.writer.flush()

    def get_loss(self):
        return np.mean(self.metrics["Loss"])

    def add_metrics(self, metric_dict):
        for metric_name in metric_dict:
            self.metrics[metric_name].append(metric_dict[metric_name])

    def update_best_model(self):
        cur_metric = self.get_metric()
        if self.best_metric < cur_metric:
            self.best_metric = cur_metric
            return True
        return False


class HeatmapMetricCounter(AbstractMetricCounter):
    def __init__(self, experiment_directory: str):
        AbstractMetricCounter.__init__(self, experiment_directory)

    def get_metric(self):
        return np.mean(self.metrics["IoU"])


class SegmentationMetricCounter(AbstractMetricCounter):
    def __init__(self, experiment_directory: str):
        AbstractMetricCounter.__init__(self, experiment_directory)

    def get_metric(self):
        return np.mean(self.metrics["IoU"])


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_metric):

        score = val_metric

        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0


def get_metric_counter(config: Dict, experiment_directory: str):
    if config["task"] == "counting":
        return HeatmapMetricCounter(experiment_directory)

    return SegmentationMetricCounter(experiment_directory)
