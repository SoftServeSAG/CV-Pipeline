import logging
import os
import os.path as osp
from datetime import datetime

import torch
import yaml
from glog import logger
from tqdm import tqdm

from model_training.common.adapters import get_model_adapter
from model_training.common.losses.loss import get_loss
from model_training.common.optimizers import get_optimizer
from model_training.common.schedulers import get_scheduler
from model_training.common.metric_utils import get_metric_counter, EarlyStopping
from model_training.common.models.semantic_segmentation.networks import get_net

from model_training.common.validator import validate_config
from model_training.common.monitors import TrainingMonitor


class Trainer(object):
    def __init__(self, config, train, val):
        validate_config(config)

        self.config = config
        self.model = get_net(config["model"])
        self.train_dataset = train
        self.val_dataset = val
        self.warmup_epochs = config.get("warmup_num", 0)

        # build experiment directory name
        self.experiment_base_dir = "_".join(
            map(
                str,
                [
                    "experiment",
                    self.config["experiment"]["folder"],
                    self.config["experiment"]["name"],
                    self.config["train"]["transform"]["size"],
                    self.config["model"]["arch"],
                    self.config["num_epochs"],
                    self.config["batch_size"],
                    str(int(datetime.timestamp(datetime.now()))),
                ],
            )
        )

        # create experiment directory
        os.makedirs(self.experiment_base_dir, exist_ok=True)

        self.metric_counter = get_metric_counter(config, self.experiment_base_dir)
        self.steps_per_epoch = config.get("steps_per_epoch", len(self.train_dataset))
        self.validation_steps = config.get("validation_steps", len(self.val_dataset))
        self.sample_input_shape = config.get("sample_input_shape", (1, 3, 32, 32))

        self.criterion = get_loss(self.config["model"]["loss"])
        self.optimizer = get_optimizer(
            config=config,
            params=filter(lambda p: p.requires_grad, self.model.parameters()),
        )
        self.scheduler = get_scheduler(config, self.optimizer)
        self.early_stopping = EarlyStopping(patience=self.config["early_stopping"])
        self.model_adapter = get_model_adapter(self.config["model"])
        self.monitor = TrainingMonitor(
            config["training_monitor"]["method"], config["training_monitor"]["interval"]
        )

        # setup logging to log file
        logger.addHandler(
            logging.FileHandler(self.experiment_base_dir + "/experiment.log")
        )

        # copy config file to experiment directory
        with open(osp.join(self.experiment_base_dir, "config.yaml"), "w") as outfile:
            yaml.dump(self.config, outfile, default_flow_style=False)

    def train(self):
        self.monitor.reset()
        for epoch in range(0, self.config["num_epochs"]):
            self.monitor.update()
            # if (epoch == self.warmup_epochs) and not (self.warmup_epochs == 0):
            #     self.model.module.unfreeze()
            #     self.optimizer = self._get_optim(self.model.parameters())
            #     self.scheduler = self._get_scheduler(self.optimizer)

            self._train_epoch(epoch)

            logger.info("Validation ...")
            self._validate(epoch)
            logger.info("Validation finished.")

            logger.info("Updating scheduler ...")
            self._update_scheduler()
            logger.info("Scheduler updated.")

            logger.info("Saving checkpoint ...")
            self._save_checkpoint(epoch)
            logger.info("Checkpoint saved.")

            self.early_stopping(val_metric=self.metric_counter.get_metric())
            if self.early_stopping.early_stop:
                logger.info("Early stopping executed.")
                break

    def _write_checkpoint(self, file_prefix: str, epoch: int):
        export_model, jit_model = self.model_adapter.get_model_export(
            self.model, self.sample_input_shape
        )

        torch.save(
            {
                "epoch": epoch,
                "optimizer": self.optimizer.state_dict(),
                "loss": self.config["model"]["loss"],
                "model": export_model,
                "scheduler": self.scheduler,
            },
            osp.join(self.experiment_base_dir, "{}.h5".format(file_prefix)),
        )

        torch.jit.save(
            jit_model, osp.join(self.experiment_base_dir, "{}.pt".format(file_prefix))
        )

    def _save_checkpoint(self, epoch: int):
        # update checkpoint
        if self.monitor.should_save_checkpoint():
            self._write_checkpoint("checkpoint_{}".format(epoch), epoch)
            self.monitor.reset()

        # update best model
        if self.metric_counter.update_best_model():
            self._write_checkpoint("best", epoch)
            logger.info(
                "Best model updated. Loss: {}".format(
                    self.metric_counter.loss_message()
                )
            )

        # save last model
        self._write_checkpoint("last", epoch)
        logger.info(
            "Last model saved. Loss: {}".format(self.metric_counter.loss_message())
        )

    def _train_epoch(self, epoch: int):
        # switch model to train mode
        self.model.train()

        self.metric_counter.clear()
        lr = self.optimizer.param_groups[0]["lr"]
        for i, data in enumerate(
            tqdm(
                self.train_dataset,
                desc="Epoch: {}, lr: {}".format(epoch, lr),
                postfix=self.metric_counter.loss_message(),
            )
        ):
            images, targets = self.model_adapter.get_input(data)
            outputs = self.model(images)

            self.optimizer.zero_grad()
            loss = self.criterion(outputs, targets)

            total_loss, loss_dict = self.model_adapter.get_loss(loss)
            total_loss.backward()
            self.optimizer.step()
            self.metric_counter.add_losses(loss_dict)

            if i >= self.steps_per_epoch:
                logger.info(
                    "Steps per epoch reach max={}, breaking training.".format(
                        self.steps_per_epoch
                    )
                )
                break

        self.metric_counter.write_to_tensorboard(epoch)
        logger.info(
            "Mean loss: {} for epoch: {}".format(self.metric_counter.get_loss(), epoch)
        )

    def _validate(self, epoch: int):
        # switch model to eval mode
        self.model.eval()

        with torch.no_grad():
            self.metric_counter.clear()

            for i, data in enumerate(
                tqdm(self.val_dataset, postfix=self.metric_counter.loss_message())
            ):
                images, targets = self.model_adapter.get_input(data)
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
                _, loss_dict = self.model_adapter.get_loss(loss)
                self.metric_counter.add_losses(loss_dict)

                # calculate metrics
                metrics = self.model_adapter.get_metrics(outputs, targets)
                self.metric_counter.add_metrics(metrics)

                if i >= self.validation_steps:
                    logger.info(
                        "Validation steps reach max={}, breaking validation.".format(
                            self.validation_steps
                        )
                    )
                    break

            self.metric_counter.write_to_tensorboard(epoch, validation=True)

    def _update_scheduler(self):
        if self.config["scheduler"]["name"] == "plateau":
            self.scheduler.step(self.metric_counter.get_metric())
        else:
            self.scheduler.step()
