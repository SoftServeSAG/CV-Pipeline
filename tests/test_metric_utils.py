import os
import shutil
from glob import glob

import pytest

from model_training.common.metric_utils import SegmentationMetricCounter


pytestmark = pytest.mark.metric_utils


@pytest.fixture(scope="module")
def segmentation_metric_counter():
    dir = "segmentation_test_dir"
    os.makedirs(dir)

    smc = SegmentationMetricCounter(dir)
    smc.add_losses({"loss": 0.8})
    smc.write_to_tensorboard(epoch_num=0)

    yield smc

    # teardown
    shutil.rmtree(dir)


def test_if_mean_loss_changes(segmentation_metric_counter: SegmentationMetricCounter):
    assert segmentation_metric_counter.get_loss() == 0.8

    segmentation_metric_counter.add_losses({"loss": 0.2})

    assert segmentation_metric_counter.get_loss() == 0.5


def test_if_writing_to_tensorboard_log_file(
    segmentation_metric_counter: SegmentationMetricCounter
):
    print(segmentation_metric_counter.writer.logdir)

    tensorboard_log_file = glob(
        segmentation_metric_counter.writer.logdir + "/events.out*"
    )[0]

    # assert if tensorboard log file exists
    assert tensorboard_log_file is not None

    with open(tensorboard_log_file, "rb") as f:
        content = f.read()

    # test it tensorboard log file not empty
    assert content != b""


def test_if_correct_loss_returned(
    segmentation_metric_counter: SegmentationMetricCounter
):
    assert segmentation_metric_counter.loss_message() == "Loss=0.5000"
