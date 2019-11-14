import cv2
import torch.backends.cudnn as cudnn
import yaml
from absl import flags, app
from glog import logger
from torch.utils.data import DataLoader

from model_training.common.trainer import Trainer
from model_training.common.validator import validate_config
from model_training.semantic_segmentation.dataset import PairedDataset

FLAGS = flags.FLAGS

flags.DEFINE_string(
    "config",
    "model_training/semantic_segmentation/config/train.yaml.default",
    "Path to config.",
)

cv2.setNumThreads(0)
cudnn.benchmark = True

logger.setLevel("INFO")


def main(_):
    with open(FLAGS.config, "r") as f:
        config = yaml.safe_load(f)

    validate_config(config)

    train = DataLoader(
        PairedDataset.from_config(config["train"]),
        config["batch_size"],
        shuffle=True,
        drop_last=True,
    )

    val = DataLoader(
        PairedDataset.from_config(config["val"]),
        config["batch_size"],
        shuffle=True,
        drop_last=True,
    )

    Trainer(config, train=train, val=val).train()


if __name__ == "__main__":
    app.run(main)
