from typing import Dict, List, Iterable

from model_training.common.aug import (
    augmentations,
    normalization,
    output_format,
    available_size_augmentations,
)

from model_training.common import consts

from model_training.common.models.semantic_segmentation.networks import (
    available_architectures,
)

from model_training.common.losses.loss import losses
from model_training.common.optimizers import available_optimizers
from model_training.common.schedulers import available_schedulers
from model_training.common.monitors import TrainingMonitor


class ConfigError(Exception):
    def __init__(self, value: List[str]):
        self.value = value

    def __str__(self):
        lines = ["\nFound errors in given config file:"]
        lines.extend(list(map(lambda error: "\t{}".format(error), self.value)))
        return "\n".join(lines)


def validate_config(config: Dict):
    config_errors = get_config_errors(config)
    if config_errors:
        raise ConfigError(config_errors)


def _concat_path(path_elements: List[str]) -> str:
    return " in ".join(map(lambda parent: "`{}`".format(parent), path_elements))


def get_config_errors(config: Dict) -> List[str]:
    found_errors = []

    if not config.keys():
        found_errors.append(consts.EMPTY_CONFIG_FILE)

    found_errors.extend(validate_missing_and_empty(config, consts.PROJECT_KEY))
    found_errors.extend(
        validate_missing_and_empty(config, consts.EXPERIMENT_DESCRIPTION_KEY)
    )

    found_errors.extend(_validate_experiment(config))
    found_errors.extend(_validate_dataset_config(config, consts.TRAIN_KEY))
    found_errors.extend(_validate_dataset_config(config, consts.VAL_KEY))
    found_errors.extend(_validate_training_monitor(config))

    found_errors.extend(validate_missing_and_empty(config, consts.TASK_KEY))
    found_errors.extend(_validate_phase(config))
    found_errors.extend(validate_missing_and_empty_int(config, consts.WARMUP_NUM_KEY))
    found_errors.extend(_validate_model(config))
    found_errors.extend(validate_missing_and_empty_int(config, consts.NUM_EPOCHS_KEY))
    found_errors.extend(validate_missing_and_empty_int(config, consts.BATCH_SIZE_KEY))
    found_errors.extend(_validate_optimizer(config))
    found_errors.extend(validate_scheduler(config))
    found_errors.extend(validate_missing_and_empty_int(config, consts.BATCH_SIZE_KEY))

    if not validate_missing(config, consts.EARLY_STOPPING_KEY):
        found_errors.extend(
            validate_missing_and_empty_int(config, consts.EARLY_STOPPING_KEY)
        )

    return found_errors


def validate_missing_and_empty_int(config: Dict, key: str) -> List[str]:
    errors = validate_missing(config, key)
    if not errors:
        errors.extend(validate_empty(config, key))

        if not errors:
            errors.extend(validate_int_value(config, key))

    return errors


def validate_same_values(
    value1, value1_path: List[str], value2, value2_path: List[str]
) -> List[str]:
    if value1 != value2:
        return [
            consts.VALUES_SHOULD_BE_THE_SAME.format(
                _concat_path(value1_path), _concat_path(value2_path)
            )
        ]

    return []


def validate_missing(config: Dict, key: str):
    try:
        config[key]
    except KeyError:
        return [consts.KEY_MISSING.format(key)]

    return []


def validate_empty(config: Dict, key: str):
    value = config[key]
    if value is None or str(value).strip() == "":
        return [consts.KEY_EMPTY.format(key)]
    else:
        return []


def validate_missing_and_empty(config: Dict, key: str):
    errors = validate_missing(config, key)

    if not errors:
        errors.extend(validate_empty(config, key))

    return errors


def validate_empty_child(config: Dict, key: str, parents: List[str]):
    value = config[key]
    if value is None or str(value).strip() == "":
        return [consts.KEY_EMPTY_WITH_PARENT.format(key, _concat_path(parents))]
    else:
        return []


def validate_missing_child(config: Dict, key: str, parents: List[str]):
    try:
        config[key]
    except KeyError:
        return [consts.KEY_MISSING_WITH_PARENT.format(key, _concat_path(parents))]

    return []


def validate_missing_and_empty_child(config: Dict, key: str, parents: List[str]):
    errors = validate_missing_child(config, key, parents)

    if not errors:
        errors.extend(validate_empty_child(config, key, parents))

    return errors


def validate_int_value(config: Dict, key: str) -> List[str]:
    if not isinstance(config[key], int):
        return [consts.NOT_INT_VALUE.format(key)]
    return []


def validate_int_value_child(config: Dict, key: str, parents: List[str]) -> List[str]:
    if not isinstance(config[key], int):
        return [consts.NOT_INT_VALUE_WITH_PARENT.format(key, _concat_path(parents))]
    return []


def validate_float_value_child(config: Dict, key: str, parents: List[str]) -> List[str]:
    if not isinstance(config[key], (float, int)):
        return [consts.NOT_FLOAT_VALUE_WITH_PARENT.format(key, _concat_path(parents))]
    return []


def validate_value_in_collection(
    config: Dict, key: str, parents: List[str], collection: Iterable
) -> List[str]:
    value = config[key]
    if value not in collection:
        return [
            consts.VALUE_NOT_IN_COLLECTION.format(
                key, _concat_path(parents), list(collection)
            )
        ]

    return []


def _validate_experiment(config: Dict):
    errors = validate_missing(config, consts.EXPERIMENT_KEY)

    if not errors:
        # folder
        errors.extend(
            validate_missing_and_empty_child(
                config[consts.EXPERIMENT_KEY],
                consts.FOLDER_KEY,
                [consts.EXPERIMENT_KEY],
            )
        )

        # name
        errors.extend(
            validate_missing_and_empty_child(
                config[consts.EXPERIMENT_KEY], consts.NAME_KEY, [consts.EXPERIMENT_KEY]
            )
        )
    return errors


def _validate_dataset_config(config: Dict, dataset_key: str):
    def missing_in_dataset(key: str) -> List[str]:
        return validate_missing_child(config[dataset_key], key, [dataset_key])

    def missing_and_empty_in_dataset(key: str) -> List[str]:
        return validate_missing_and_empty_child(config[dataset_key], key, [dataset_key])

    def missing_and_empty_in_transform(key: str) -> List[str]:
        return validate_missing_and_empty_child(
            config[dataset_key][consts.TRANSFORM_KEY],
            key,
            [consts.TRANSFORM_KEY, dataset_key],
        )

    def validate_int_in_transform(key: str) -> List[str]:
        return validate_int_value_child(
            config[dataset_key][consts.TRANSFORM_KEY],
            key,
            [consts.TRANSFORM_KEY, dataset_key],
        )

    def value_in_collection_in_transform(key: str, collection: Iterable) -> List[str]:
        return validate_value_in_collection(
            config[dataset_key][consts.TRANSFORM_KEY],
            key,
            [consts.TRANSFORM_KEY, dataset_key],
            collection,
        )

    errors = validate_missing(config, dataset_key)

    if not errors:
        # files_a
        errors.extend(missing_and_empty_in_dataset(consts.FILES_A_KEY))

        # files_b
        errors.extend(missing_and_empty_in_dataset(consts.FILES_B_KEY))

        # transform
        transform_missing_errors = missing_in_dataset(consts.TRANSFORM_KEY)
        errors.extend(transform_missing_errors)

        if not transform_missing_errors:
            # size
            size_missing_and_empty = missing_and_empty_in_transform(consts.SIZE_KEY)
            errors.extend(size_missing_and_empty)

            if not size_missing_and_empty:
                errors.extend(validate_int_in_transform(consts.SIZE_KEY))

            # augmentation_scope
            augmentation_scope_missing_and_empty = missing_and_empty_in_transform(
                consts.AUGMENTATION_SCOPE_KEY
            )
            errors.extend(augmentation_scope_missing_and_empty)

            if not augmentation_scope_missing_and_empty:
                errors.extend(
                    value_in_collection_in_transform(
                        consts.AUGMENTATION_SCOPE_KEY, augmentations.keys()
                    )
                )

            # images_normalization
            images_normalization_missing_and_empty = missing_and_empty_in_transform(
                consts.IMAGE_NORMALIZATION_KEY
            )
            errors.extend(images_normalization_missing_and_empty)

            if not images_normalization_missing_and_empty:
                errors.extend(
                    value_in_collection_in_transform(
                        consts.IMAGE_NORMALIZATION_KEY, normalization.keys()
                    )
                )

            # images_output_format_type
            images_output_format_type_missing_and_empty = missing_and_empty_in_transform(
                consts.IMAGES_OUTPUT_FORMAT_TYPE_KEY
            )
            errors.extend(images_output_format_type_missing_and_empty)

            if not images_output_format_type_missing_and_empty:
                errors.extend(
                    value_in_collection_in_transform(
                        consts.IMAGES_OUTPUT_FORMAT_TYPE_KEY, output_format.keys()
                    )
                )

            # masks_normalization
            masks_normalization_missing_and_empty = missing_and_empty_in_transform(
                consts.MASK_NORMALIZATION_KEY
            )
            errors.extend(masks_normalization_missing_and_empty)

            if not masks_normalization_missing_and_empty:
                errors.extend(
                    value_in_collection_in_transform(
                        consts.MASK_NORMALIZATION_KEY, normalization.keys()
                    )
                )

            # masks_output_format_type
            masks_output_format_type_missing_and_empty = missing_and_empty_in_transform(
                consts.MASKS_OUTPUT_FORMAT_TYPE_KEY
            )
            errors.extend(masks_output_format_type_missing_and_empty)

            if not masks_output_format_type_missing_and_empty:
                errors.extend(
                    value_in_collection_in_transform(
                        consts.MASKS_OUTPUT_FORMAT_TYPE_KEY, output_format.keys()
                    )
                )

            # size_transform
            size_transform_missing_and_empty = missing_and_empty_in_transform(
                consts.SIZE_TRANSFORM_KEY
            )
            errors.extend(size_transform_missing_and_empty)

            if not size_transform_missing_and_empty:
                errors.extend(
                    value_in_collection_in_transform(
                        consts.SIZE_TRANSFORM_KEY, available_size_augmentations
                    )
                )

        # norm
        norm_missing_and_empty = missing_and_empty_in_dataset(consts.NORM_KEY)
        errors.extend(norm_missing_and_empty)

        # preload
        preload_missing_and_empty = missing_and_empty_in_dataset(consts.PRELOAD_KEY)
        errors.extend(preload_missing_and_empty)

        # preload_size
        preload_size_missing_and_empty = missing_and_empty_in_dataset(
            consts.PRELOAD_SIZE_KEY
        )
        errors.extend(preload_size_missing_and_empty)

        # if preload size not missing lets check if it has integer value
        if not preload_size_missing_and_empty:
            errors.extend(
                validate_int_value_child(
                    config[dataset_key], consts.PRELOAD_SIZE_KEY, [dataset_key]
                )
            )

        # bounds
        bounds_missing_and_empty = missing_and_empty_in_dataset(consts.BOUNDS_KEY)
        errors.extend(bounds_missing_and_empty)

        def is_string(value) -> bool:
            return isinstance(value, str)

        def has_more_than_two_values(value) -> bool:
            return len(list(iter(value))) > 2

        def is_any_value_not_a_number(value) -> bool:
            return any(
                not isinstance(value, (float, int)) for value in list(set(value))
            )

        def are_list_values_between_min_and_max(value: List, min_val, max_val) -> bool:
            return max(value) > max_val or min(value) < min_val

        # if bounds not missing, lets test its values
        if not bounds_missing_and_empty:
            bounds = config[dataset_key][consts.BOUNDS_KEY]
            min_value = 0
            max_value = 1
            if (
                is_string(bounds)
                or has_more_than_two_values(bounds)
                or is_any_value_not_a_number(bounds)
                or are_list_values_between_min_and_max(bounds, min_value, max_value)
            ):
                errors.append(
                    consts.VALUE_NOT_COLLECTION_OF_NUMBERS_WITH_LENGTH.format(
                        consts.BOUNDS_KEY,
                        _concat_path([dataset_key]),
                        "[{}, {}]".format(min_value, max_value),
                        2,
                    )
                )

    return errors


def _validate_training_monitor(config: Dict):
    errors = validate_missing(config, consts.TRAINING_MONITOR_KEY)

    if not errors:

        missing_or_empty_method = validate_missing_and_empty_child(
            config[consts.TRAINING_MONITOR_KEY],
            consts.METHOD_KEY,
            [consts.TRAINING_MONITOR_KEY],
        )
        errors.extend(missing_or_empty_method)
        if not missing_or_empty_method:
            errors.extend(
                validate_value_in_collection(
                    config[consts.TRAINING_MONITOR_KEY],
                    consts.METHOD_KEY,
                    [consts.TRAINING_MONITOR_KEY],
                    TrainingMonitor.available_methods,
                )
            )

        missing_or_empty_interval = validate_missing_and_empty_child(
            config[consts.TRAINING_MONITOR_KEY],
            consts.INTERVAL_KEY,
            [consts.TRAINING_MONITOR_KEY],
        )
        errors.extend(missing_or_empty_interval)
        if not missing_or_empty_interval:
            errors.extend(
                validate_int_value_child(
                    config[consts.TRAINING_MONITOR_KEY],
                    consts.INTERVAL_KEY,
                    [consts.TRAINING_MONITOR_KEY],
                )
            )

    return errors


def _validate_phase(config: Dict) -> List[str]:
    errors = []
    if not validate_missing(config, consts.PHASE_KEY):
        errors.extend(validate_empty(config, consts.PHASE_KEY))

    return errors


def _validate_model(config: Dict) -> List[str]:
    errors = validate_missing(config, consts.MODEL_KEY)

    if not errors:
        # arch
        missing_arch = validate_missing_and_empty_child(
            config[consts.MODEL_KEY], consts.ARCH_KEY, [consts.MODEL_KEY]
        )
        errors.extend(missing_arch)

        if not missing_arch:
            errors.extend(
                validate_value_in_collection(
                    config[consts.MODEL_KEY],
                    consts.ARCH_KEY,
                    [consts.MODEL_KEY],
                    available_architectures,
                )
            )

        # loss
        missing_loss = validate_missing_child(
            config[consts.MODEL_KEY], consts.LOSS_KEY, [consts.MODEL_KEY]
        )
        errors.extend(missing_loss)

        if not missing_loss:
            missing_loss_name = validate_missing_and_empty_child(
                config[consts.MODEL_KEY][consts.LOSS_KEY],
                consts.NAME_KEY,
                [consts.LOSS_KEY, consts.MODEL_KEY],
            )
            errors.extend(missing_loss_name)

            if not missing_loss_name:
                errors.extend(
                    validate_value_in_collection(
                        config[consts.MODEL_KEY][consts.LOSS_KEY],
                        consts.NAME_KEY,
                        [consts.LOSS_KEY, consts.MODEL_KEY],
                        losses.keys(),
                    )
                )

        # classes
        missing_classes = validate_missing_and_empty_child(
            config[consts.MODEL_KEY], consts.CLASSES_KEY, [consts.MODEL_KEY]
        )
        errors.extend(missing_classes)
        if not missing_classes:
            errors.extend(
                validate_int_value_child(
                    config[consts.MODEL_KEY], consts.CLASSES_KEY, [consts.MODEL_KEY]
                )
            )

        # pretrained
        missing_pretrained = validate_missing_and_empty_child(
            config[consts.MODEL_KEY], consts.PRETRAINED_KEY, [consts.MODEL_KEY]
        )
        errors.extend(missing_pretrained)

        # pretrained_weights_path
        if not missing_pretrained and config[consts.MODEL_KEY][consts.PRETRAINED_KEY]:
            errors.extend(
                validate_missing_and_empty_child(
                    config[consts.MODEL_KEY],
                    consts.PRETRAINED_WEIGHTS_PATH_KEY,
                    [consts.MODEL_KEY],
                )
            )

        # task
        missing_model_task = validate_missing_and_empty_child(
            config[consts.MODEL_KEY], consts.TASK_KEY, [consts.MODEL_KEY]
        )
        missing_config_task = validate_missing_and_empty(config, consts.TASK_KEY)
        errors.extend(missing_model_task)

        # validate if task from model is the same as task from config
        if not missing_model_task and not missing_config_task:
            errors.extend(
                validate_same_values(
                    config[consts.MODEL_KEY][consts.TASK_KEY],
                    [consts.TASK_KEY, consts.MODEL_KEY],
                    config[consts.TASK_KEY],
                    [consts.TASK_KEY],
                )
            )

        # norm_layer
        errors.extend(
            validate_missing_and_empty_child(
                config[consts.MODEL_KEY], consts.NORM_LAYER_KEY, [consts.MODEL_KEY]
            )
        )

    return errors


def _validate_optimizer(config: Dict) -> List[str]:
    errors = validate_missing(config, consts.OPTIMIZER_KEY)

    if not errors:
        missing_name = validate_missing_and_empty_child(
            config[consts.OPTIMIZER_KEY], consts.NAME_KEY, [consts.OPTIMIZER_KEY]
        )
        errors.extend(missing_name)
        if not missing_name:
            errors.extend(
                validate_value_in_collection(
                    config[consts.OPTIMIZER_KEY],
                    consts.NAME_KEY,
                    [consts.OPTIMIZER_KEY],
                    available_optimizers,
                )
            )

        missing_lr = validate_missing_and_empty_child(
            config[consts.OPTIMIZER_KEY], consts.LR_KEY, [consts.OPTIMIZER_KEY]
        )
        errors.extend(missing_lr)
        if not missing_lr:
            errors.extend(
                validate_float_value_child(
                    config[consts.OPTIMIZER_KEY], consts.LR_KEY, [consts.OPTIMIZER_KEY]
                )
            )

    return errors


def validate_scheduler(config: Dict) -> List[str]:
    errors = validate_missing(config, consts.SCHEDULER_KEY)

    if not errors:
        missing_name = validate_missing_and_empty_child(
            config[consts.SCHEDULER_KEY], consts.NAME_KEY, [consts.SCHEDULER_KEY]
        )
        errors.extend(missing_name)

        if not missing_name:
            invalid_name = validate_value_in_collection(
                config[consts.SCHEDULER_KEY],
                consts.NAME_KEY,
                [consts.SCHEDULER_KEY],
                available_schedulers.keys(),
            )
            errors.extend(invalid_name)
            if not invalid_name:
                for required_param in available_schedulers[
                    config[consts.SCHEDULER_KEY][consts.NAME_KEY]
                ]:
                    errors.extend(
                        validate_missing_and_empty_child(
                            config[consts.SCHEDULER_KEY],
                            required_param,
                            [consts.SCHEDULER_KEY],
                        )
                    )

    return errors
