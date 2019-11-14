import os
import shutil
from typing import Dict

import pytest
import yaml
from torch.utils.data import DataLoader

from model_training.common.trainer import Trainer
from model_training.semantic_segmentation.dataset import PairedDataset


pytestmark = pytest.mark.models


def _teardown_trainer_dir(directory: str):
    if os.path.exists(directory):
        shutil.rmtree(directory)

    # assert that experiment directory removed after test
    assert not os.path.exists(directory)


def _build_trainer(config_file: Dict):
    train = DataLoader(
        PairedDataset.from_config(config_file["train"]),
        config_file["batch_size"],
        shuffle=True,
        drop_last=False,
    )

    val = DataLoader(
        PairedDataset.from_config(config_file["val"]),
        config_file["batch_size"],
        shuffle=True,
        drop_last=False,
    )

    trainer = Trainer(config_file, train=train, val=val)
    trainer.train()
    return trainer


@pytest.fixture(scope="module")
def cv_trainer(config: Dict) -> Trainer:
    trainer = _build_trainer(config)

    yield trainer

    # # teardown for experiment directory
    _teardown_trainer_dir(trainer.experiment_base_dir)


@pytest.fixture(scope="module")
def pretrained_trainer(config: Dict, cv_trainer: Trainer) -> Trainer:
    new_config = config.copy()
    new_config["model"]["pretrained_weights_path"] = (
        cv_trainer.experiment_base_dir + "/checkpoint_0.h5"
    )
    new_config["model"]["pretrained"] = True

    new_trainer = _build_trainer(new_config)
    return new_trainer


def test_if_experiment_directory_created(cv_trainer: Trainer):
    assert os.path.exists(cv_trainer.experiment_base_dir)


def test_if_experiment_directory_contain_copied_config_file(cv_trainer: Trainer):
    assert os.path.exists(cv_trainer.experiment_base_dir + "/config.yaml")


def test_if_training_log_file_created(cv_trainer: Trainer):
    assert os.path.exists(cv_trainer.experiment_base_dir + "/experiment.log")


def test_if_best_model_saved(cv_trainer: Trainer):
    assert os.path.exists(cv_trainer.experiment_base_dir + "/best.h5")
    assert os.path.exists(cv_trainer.experiment_base_dir + "/best.pt")


def test_if_last_model_saved(cv_trainer: Trainer):
    assert os.path.exists(cv_trainer.experiment_base_dir + "/last.h5")
    assert os.path.exists(cv_trainer.experiment_base_dir + "/last.pt")


def test_if_checkpoint_after_0_epoch_saved(cv_trainer: Trainer):
    assert os.path.exists(cv_trainer.experiment_base_dir + "/checkpoint_0.h5")
    assert os.path.exists(cv_trainer.experiment_base_dir + "/checkpoint_0.pt")


def test_copied_config_content(cv_trainer: Trainer, config: Dict):
    with open(cv_trainer.experiment_base_dir + "/config.yaml", "r") as f:
        experiment_saved_config = yaml.safe_load(f)

    assert set(experiment_saved_config.keys()) == set(config.keys())

    for key in config.keys():
        assert config[key] == experiment_saved_config[key]


def test_training_from_checkpoint(pretrained_trainer: Trainer):
    assert os.path.exists(pretrained_trainer.experiment_base_dir + "/best.pt")
    assert os.path.exists(pretrained_trainer.experiment_base_dir + "/best.h5")
    assert os.path.exists(pretrained_trainer.experiment_base_dir + "/last.h5")
    assert os.path.exists(pretrained_trainer.experiment_base_dir + "/last.pt")
    assert os.path.exists(pretrained_trainer.experiment_base_dir + "/checkpoint_0.h5")
    assert os.path.exists(pretrained_trainer.experiment_base_dir + "/checkpoint_0.pt")

    _teardown_trainer_dir(pretrained_trainer.experiment_base_dir)
