import pytest
import yaml

from model_training.common.validator import get_config_errors

pytestmark = pytest.mark.validator


def test_base_correct_config(config):
    assert not get_config_errors(config)


def test_missing_values_in_config():
    errors = get_config_errors({})

    assert "Given config file is empty." in errors
    assert "`project` is missing." in errors
    assert "`experiment_desc` is missing." in errors
    assert "`experiment` is missing." in errors
    assert "`train` is missing." in errors
    assert "`val` is missing." in errors
    assert "`training_monitor` is missing." in errors
    assert "`task` is missing." in errors
    assert "`model` is missing." in errors
    assert "`warmup_num` is missing." in errors
    assert "`num_epochs` is missing." in errors
    assert "`batch_size` is missing." in errors
    assert "`optimizer` is missing." in errors
    assert "`scheduler` is missing." in errors


def test_empty_values_in_config():
    assert "`project` have empty value." in get_config_errors({"project": ""})
    assert "`experiment_desc` have empty value." in get_config_errors(
        {"experiment_desc": ""}
    )
    assert "`phase` have empty value." in get_config_errors({"phase": ""})
    assert "`task` have empty value." in get_config_errors({"task": ""})
    assert "`warmup_num` have empty value." in get_config_errors({"warmup_num": ""})
    assert "`num_epochs` have empty value." in get_config_errors({"num_epochs": ""})
    assert "`batch_size` have empty value." in get_config_errors({"batch_size": ""})
    assert "`early_stopping` have empty value." in get_config_errors(
        {"early_stopping": ""}
    )


def test_integer_values_in_config():
    warmup_num_error = "`warmup_num` should have integer value."
    assert warmup_num_error in get_config_errors({"warmup_num": "asd"})
    assert warmup_num_error in get_config_errors({"warmup_num": "1"})
    assert warmup_num_error in get_config_errors({"warmup_num": 1.0})
    assert warmup_num_error not in get_config_errors({"warmup_num": 1})

    num_epochs_error = "`num_epochs` should have integer value."
    assert num_epochs_error in get_config_errors({"num_epochs": "asd"})
    assert num_epochs_error in get_config_errors({"num_epochs": "1"})
    assert num_epochs_error in get_config_errors({"num_epochs": 1.0})
    assert num_epochs_error not in get_config_errors({"num_epochs": 1})

    batch_size_error = "`batch_size` should have integer value."
    assert batch_size_error in get_config_errors({"batch_size": "asd"})
    assert batch_size_error in get_config_errors({"batch_size": "1"})
    assert batch_size_error in get_config_errors({"batch_size": 1.0})
    assert batch_size_error not in get_config_errors({"batch_size": 1})

    batch_size_error = "`early_stopping` should have integer value."
    assert batch_size_error in get_config_errors({"early_stopping": "asd"})
    assert batch_size_error in get_config_errors({"early_stopping": "1"})
    assert batch_size_error in get_config_errors({"early_stopping": 1.0})
    assert batch_size_error not in get_config_errors({"early_stopping": 1})


def test_missing_values_in_experiment():
    errors = get_config_errors({"experiment": {}})
    assert "`folder` in `experiment` is missing." in errors
    assert "`name` in `experiment` is missing." in errors


def test_empty_values_in_experiment():
    assert "`folder` in `experiment` have empty value." in get_config_errors(
        {"experiment": {"folder": ""}}
    )
    assert "`name` in `experiment` have empty value." in get_config_errors(
        {"experiment": {"name": ""}}
    )


def test_missing_values_in_train():
    errors = get_config_errors({"train": {}})

    assert "`files_a` in `train` is missing." in errors
    assert "`files_b` in `train` is missing." in errors
    assert "`transform` in `train` is missing." in errors
    assert "`norm` in `train` is missing." in errors
    assert "`preload` in `train` is missing." in errors
    assert "`preload_size` in `train` is missing." in errors
    assert "`bounds` in `train` is missing." in errors


def test_empty_values_in_train():
    assert "`files_a` in `train` have empty value." in get_config_errors(
        {"train": {"files_a": ""}}
    )
    assert "`files_b` in `train` have empty value." in get_config_errors(
        {"train": {"files_b": ""}}
    )
    assert "`norm` in `train` have empty value." in get_config_errors(
        {"train": {"norm": ""}}
    )
    assert "`preload_size` in `train` have empty value." in get_config_errors(
        {"train": {"preload_size": ""}}
    )
    assert "`bounds` in `train` have empty value." in get_config_errors(
        {"train": {"bounds": ""}}
    )


def test_integer_values_in_train():
    errors = get_config_errors({"train": {"preload_size": "asd"}})
    assert "`preload_size` in `train` should have integer value." in errors


def test_collections_of_numbers_in_train():
    error_msg = "`bounds` in `train` should be iterable of numbers in range [0, 1] with length = 2."
    assert error_msg in get_config_errors({"train": {"bounds": "asd"}})
    assert error_msg in get_config_errors({"train": {"bounds": "[asd, asd]"}})
    assert error_msg in get_config_errors({"train": {"bounds": "[2, 1, 3]"}})
    assert error_msg in get_config_errors({"train": {"bounds": ["asd", "asd"]}})
    assert error_msg in get_config_errors({"train": {"bounds": [2, 1]}})
    assert error_msg in get_config_errors({"train": {"bounds": [2, 1, 4]}})

    assert error_msg not in get_config_errors({"train": {"bounds": [0, 1]}})
    assert error_msg not in get_config_errors({"train": {"bounds": [0.1, 0.5]}})


def test_if_missing_keys_in_transform_in_train():
    errors = get_config_errors({"train": {"transform": {}}})

    assert "`size` in `transform` in `train` is missing." in errors
    assert "`augmentation_scope` in `transform` in `train` is missing." in errors
    assert "`images_normalization` in `transform` in `train` is missing." in errors
    assert "`images_output_format_type` in `transform` in `train` is missing." in errors
    assert "`masks_normalization` in `transform` in `train` is missing." in errors
    assert "`masks_output_format_type` in `transform` in `train` is missing." in errors
    assert "`size_transform` in `transform` in `train` is missing." in errors


def test_if_required_keys_empty_in_transform_in_train():
    errors = get_config_errors(
        {
            "train": {
                "transform": {
                    "size": "",
                    "augmentation_scope": "",
                    "images_normalization": "",
                    "images_output_format_type": "",
                    "masks_normalization": "",
                    "masks_output_format_type": "",
                    "size_transform": "",
                }
            }
        }
    )

    assert "`size` in `transform` in `train` have empty value." in errors
    assert "`augmentation_scope` in `transform` in `train` have empty value." in errors
    assert (
        "`images_normalization` in `transform` in `train` have empty value." in errors
    )
    assert (
        "`images_output_format_type` in `transform` in `train` have empty value."
        in errors
    )
    assert "`masks_normalization` in `transform` in `train` have empty value." in errors
    assert (
        "`masks_output_format_type` in `transform` in `train` have empty value."
        in errors
    )
    assert "`size_transform` in `transform` in `train` have empty value." in errors


def test_required_integer_values_in_transform_in_train():
    errors = get_config_errors({"train": {"transform": {"size": "asd"}}})
    assert "`size` in `transform` in `train` should have integer value." in errors


def test_required_values_from_collections_in_transform_train():
    errors = get_config_errors(
        {
            "train": {
                "transform": {
                    "augmentation_scope": "this_is_not_correct_value",
                    "images_normalization": "this_is_not_correct_value",
                    "images_output_format_type": "this_is_not_correct_value",
                    "masks_normalization": "this_is_not_correct_value",
                    "masks_output_format_type": "this_is_not_correct_value",
                    "size_transform": "this_is_not_correct_value",
                }
            }
        }
    )
    assert (
        "`augmentation_scope` in `transform` in `train` should have one of values: "
        + "['strong', 'weak', 'none', 'geometric']."
    ) in errors
    assert (
        "`images_normalization` in `transform` in `train` should have one of values: "
        + "['none', 'default', 'div255']."
    ) in errors
    assert (
        "`images_output_format_type` in `transform` in `train` should have one of values: "
        + "['none', 'float', 'long']."
    ) in errors
    assert (
        "`masks_normalization` in `transform` in `train` should have one of values: "
        + "['none', 'default', 'div255']."
    ) in errors
    assert (
        "`masks_output_format_type` in `transform` in `train` should have one of values: "
        + "['none', 'float', 'long']."
    ) in errors
    assert (
        "`size_transform` in `transform` in `train` should have one of values: "
        + "['none', 'resize', 'random', 'center']."
    ) in errors


def test_missing_values_in_val():
    errors = get_config_errors({"val": {}})

    assert "`files_a` in `val` is missing." in errors
    assert "`files_b` in `val` is missing." in errors
    assert "`transform` in `val` is missing." in errors
    assert "`norm` in `val` is missing." in errors
    assert "`preload` in `val` is missing." in errors
    assert "`preload_size` in `val` is missing." in errors
    assert "`bounds` in `val` is missing." in errors


def test_empty_values_in_val():
    assert "`files_a` in `val` have empty value." in get_config_errors(
        {"val": {"files_a": ""}}
    )
    assert "`files_b` in `val` have empty value." in get_config_errors(
        {"val": {"files_b": ""}}
    )
    assert "`norm` in `val` have empty value." in get_config_errors(
        {"val": {"norm": ""}}
    )
    assert "`preload_size` in `val` have empty value." in get_config_errors(
        {"val": {"preload_size": ""}}
    )
    assert "`bounds` in `val` have empty value." in get_config_errors(
        {"val": {"bounds": ""}}
    )


def test_collections_of_numbers_in_val():
    error_msg = "`bounds` in `val` should be iterable of numbers in range [0, 1] with length = 2."
    assert error_msg in get_config_errors({"val": {"bounds": "asd"}})
    assert error_msg in get_config_errors({"val": {"bounds": "[asd, asd]"}})
    assert error_msg in get_config_errors({"val": {"bounds": "[2, 1, 3]"}})
    assert error_msg in get_config_errors({"val": {"bounds": ["asd", "asd"]}})
    assert error_msg in get_config_errors({"val": {"bounds": [2, 1]}})
    assert error_msg in get_config_errors({"val": {"bounds": [2, 1, 4]}})

    assert error_msg not in get_config_errors({"val": {"bounds": [0, 1]}})
    assert error_msg not in get_config_errors({"val": {"bounds": [0.1, 0.5]}})


def test_integer_values_in_val():
    errors = get_config_errors({"val": {"preload_size": "asd"}})
    assert "`preload_size` in `val` should have integer value." in errors


def test_if_missing_keys_in_transform_in_val():
    errors = get_config_errors({"val": {"transform": {}}})

    assert "`size` in `transform` in `val` is missing." in errors
    assert "`augmentation_scope` in `transform` in `val` is missing." in errors
    assert "`images_normalization` in `transform` in `val` is missing." in errors
    assert "`images_output_format_type` in `transform` in `val` is missing." in errors
    assert "`masks_normalization` in `transform` in `val` is missing." in errors
    assert "`masks_output_format_type` in `transform` in `val` is missing." in errors
    assert "`size_transform` in `transform` in `val` is missing." in errors


def test_if_required_keys_empty_in_transform_in_val():
    errors = get_config_errors(
        {
            "val": {
                "transform": {
                    "size": "",
                    "augmentation_scope": "",
                    "images_normalization": "",
                    "images_output_format_type": "",
                    "masks_normalization": "",
                    "masks_output_format_type": "",
                    "size_transform": "",
                }
            }
        }
    )

    assert "`size` in `transform` in `val` have empty value." in errors
    assert "`augmentation_scope` in `transform` in `val` have empty value." in errors
    assert "`images_normalization` in `transform` in `val` have empty value." in errors
    assert (
        "`images_output_format_type` in `transform` in `val` have empty value."
        in errors
    )
    assert "`masks_normalization` in `transform` in `val` have empty value." in errors
    assert (
        "`masks_output_format_type` in `transform` in `val` have empty value." in errors
    )
    assert "`size_transform` in `transform` in `val` have empty value." in errors


def test_required_integer_values_in_transform_in_val():
    errors = get_config_errors({"val": {"transform": {"size": "asd"}}})
    assert "`size` in `transform` in `val` should have integer value." in errors


def test_training_monitor():
    assert "`method` in `training_monitor` is missing." in get_config_errors(
        {"training_monitor": {}}
    )
    assert "`interval` in `training_monitor` is missing." in get_config_errors(
        {"training_monitor": {}}
    )

    assert "`method` in `training_monitor` have empty value." in get_config_errors(
        {"training_monitor": {"method": ""}}
    )
    assert "`interval` in `training_monitor` have empty value." in get_config_errors(
        {"training_monitor": {"interval": ""}}
    )

    method_error = (
        "`method` in `training_monitor` should have one of values: ['time', 'epochs']."
    )
    assert method_error in get_config_errors(
        {"training_monitor": {"method": "not_correct_value"}}
    )
    assert method_error not in get_config_errors(
        {"training_monitor": {"method": "time"}}
    )
    assert method_error not in get_config_errors(
        {"training_monitor": {"method": "epochs"}}
    )

    interval_error = "`interval` in `training_monitor` should have integer value."
    assert interval_error in get_config_errors(
        {"training_monitor": {"interval": "asd"}}
    )
    assert interval_error in get_config_errors({"training_monitor": {"interval": 1.0}})
    assert interval_error not in get_config_errors(
        {"training_monitor": {"interval": 1}}
    )


def test_missing_values_in_models():
    errors = get_config_errors({"model": {}})
    assert "`arch` in `model` is missing." in errors
    assert "`loss` in `model` is missing." in errors
    assert "`classes` in `model` is missing." in errors
    assert "`pretrained` in `model` is missing." in errors
    assert "`pretrained_weights_path` in `model` is missing." in get_config_errors(
        {"model": {"pretrained": True}}
    )
    assert "`pretrained_weights_path` in `model` is missing." not in get_config_errors(
        {"model": {"pretrained": False}}
    )
    assert "`task` in `model` is missing." in errors
    assert "`norm_layer` in `model` is missing." in errors


def test_empty_values_in_models():
    assert "`arch` in `model` have empty value." in get_config_errors(
        {"model": {"arch": ""}}
    )
    assert "`classes` in `model` have empty value." in get_config_errors(
        {"model": {"classes": ""}}
    )
    assert "`pretrained` in `model` have empty value." in get_config_errors(
        {"model": {"pretrained": ""}}
    )
    assert (
        "`pretrained_weights_path` in `model` have empty value."
        in get_config_errors(
            {"model": {"pretrained": True, "pretrained_weights_path": ""}}
        )
    )
    assert (
        "`pretrained_weights_path` in `model` have empty value."
        not in get_config_errors(
            {"model": {"pretrained": False, "pretrained_weights_path": ""}}
        )
    )
    assert "`task` in `model` have empty value." in get_config_errors(
        {"model": {"task": ""}}
    )
    assert "`norm_layer` in `model` have empty value." in get_config_errors(
        {"model": {"norm_layer": ""}}
    )


def test_if_task_from_model_is_the_same_as_from_config():
    assert (
        "Values in `task` in `model` and 'task' should be the same."
        not in get_config_errors(
            {"task": "task_name", "model": {"task": "different_task_name"}}
        )
    )
    assert (
        "Values in `task` in `model` and 'task' should be the same."
        not in get_config_errors({"task": "task_name", "model": {"task": "task_name"}})
    )
    assert (
        "Values in `task` in `model` and 'task' should be the same."
        not in get_config_errors({"task": "", "model": {"task": "task_name"}})
    )
    assert (
        "Values in `task` in `model` and 'task' should be the same."
        not in get_config_errors({"task": "", "model": {"task": ""}})
    )
    assert (
        "Values in `task` in `model` and 'task' should be the same."
        not in get_config_errors({"task": "", "model": {}})
    )


def test_integer_values_in_models():
    error = "`classes` in `model` should have integer value."
    assert error in get_config_errors({"model": {"classes": "asd"}})
    assert error in get_config_errors({"model": {"classes": "2"}})
    assert error in get_config_errors({"model": {"classes": 2.0}})

    assert error not in get_config_errors({"model": {"classes": 2}})


def test_architecture_value_in_model():

    error = (
        "`arch` in `model` should have one of values: "
        + "['fpn_mobilenet', 'fpn_inception', 'unet_seresnext', 'unet_densenet', 'unet_resnet', 'espnet']."
    )

    assert error in get_config_errors({"model": {"arch": "not_available_architecture"}})
    assert error not in get_config_errors({"model": {"arch": "fpn_mobilenet"}})
    assert error not in get_config_errors({"model": {"arch": "espnet"}})


def test_name_value_in_loss_in_model():
    assert "`name` in `loss` in `model` is missing." in get_config_errors(
        {"model": {"loss": {}}}
    )
    assert "`name` in `loss` in `model` have empty value." in get_config_errors(
        {"model": {"loss": {"name": ""}}}
    )


def test_missing_values_in_optimizer():
    errors = get_config_errors({"optimizer": {}})
    assert "`name` in `optimizer` is missing." in errors
    assert "`lr` in `optimizer` is missing." in errors


def test_float_values_in_optimizer():
    error = "`lr` in `optimizer` should have float value."

    assert error in get_config_errors({"optimizer": {"lr": "asd"}})
    assert error in get_config_errors({"optimizer": {"lr": "1"}})
    assert error in get_config_errors({"optimizer": {"lr": "1.0"}})
    assert error in get_config_errors({"optimizer": {"lr": "0.1"}})
    assert error not in get_config_errors({"optimizer": {"lr": 1.0}})
    assert error not in get_config_errors({"optimizer": {"lr": 0.01}})
    assert error not in get_config_errors({"optimizer": {"lr": 1}})


def test_empty_values_in_optimizer():
    errors = get_config_errors({"optimizer": {"name": "", "lr": ""}})
    assert "`name` in `optimizer` have empty value." in errors
    assert "`lr` in `optimizer` have empty value." in errors


def test_name_value_in_optimizer():
    error = (
        "`name` in `optimizer` should have one of values: ['adam', 'sgd', 'adadelta']."
    )

    assert error in get_config_errors(
        {"optimizer": {"name": "not_available_architecture"}}
    )
    assert error not in get_config_errors({"optimizer": {"name": "adam"}})
    assert error not in get_config_errors({"optimizer": {"name": "sgd"}})
    assert error not in get_config_errors({"optimizer": {"name": "adadelta"}})


def test_missing_name_in_scheduler():
    errors = get_config_errors({"scheduler": {}})
    assert "`name` in `scheduler` is missing." in errors


def test_missing_values_for_plateau_in_scheduler():
    errors = get_config_errors({"scheduler": {"name": "plateau"}})

    assert "`mode` in `scheduler` is missing." in errors
    assert "`patience` in `scheduler` is missing." in errors
    assert "`factor` in `scheduler` is missing." in errors
    assert "`min_lr` in `scheduler` is missing." in errors


def test_empty_values_for_plateau_in_scheduler():
    errors = get_config_errors(
        {
            "scheduler": {
                "name": "plateau",
                "mode": "",
                "patience": "",
                "factor": "",
                "min_lr": "",
            }
        }
    )
    assert "`mode` in `scheduler` have empty value." in errors
    assert "`patience` in `scheduler` have empty value." in errors
    assert "`factor` in `scheduler` have empty value." in errors
    assert "`min_lr` in `scheduler` have empty value." in errors


def test_missing_values_for_MultiStepLR_in_scheduler():
    errors = get_config_errors({"scheduler": {"name": "MultiStepLR"}})

    assert "`milestones` in `scheduler` is missing." in errors
    assert "`gamma` in `scheduler` is missing." in errors


def test_empty_values_for_MultiStepLR_in_scheduler():
    errors = get_config_errors(
        {"scheduler": {"name": "MultiStepLR", "milestones": "", "gamma": ""}}
    )
    assert "`milestones` in `scheduler` have empty value." in errors
    assert "`gamma` in `scheduler` have empty value." in errors


def test_name_in_collections_in_scheduler():
    assert (
        "`name` in `scheduler` should have one of values: ['plateau', 'MultiStepLR']."
        in get_config_errors({"scheduler": {"name": "not_available_architecture"}})
    )
    assert (
        "`name` in `scheduler` should have one of values: ['plateau', 'MultiStepLR']."
        not in get_config_errors({"scheduler": {"name": "plateau"}})
    )
    assert (
        "`name` in `scheduler` should have one of values: ['plateau', 'MultiStepLR']."
        not in get_config_errors({"scheduler": {"name": "MultiStepLR"}})
    )
