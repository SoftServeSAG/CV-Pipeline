from typing import Dict

import albumentations as albu
import numpy as np
import torch

output_format = {
    "none": lambda array: array,
    "float": lambda array: torch.FloatTensor(array),
    "long": lambda array: torch.LongTensor(array),
}

normalization = {
    "none": lambda array: array,
    "default": lambda array: albu.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )(image=array)["image"],
    "div255": lambda array: array / 255,
}

augmentations = {
    "strong": albu.Compose(
        [
            albu.Flip(),
            albu.ShiftScaleRotate(
                shift_limit=0.0, scale_limit=0.2, rotate_limit=30, p=0.4
            ),
            albu.OpticalDistortion(),
            albu.ElasticTransform(),
            albu.GaussNoise(),
            albu.RandomShadow(),
            albu.OneOf(
                [
                    albu.CLAHE(clip_limit=2),
                    albu.IAASharpen(),
                    albu.IAAEmboss(),
                    albu.RandomBrightnessContrast(),
                    albu.RandomGamma(),
                    albu.MedianBlur(),
                ],
                p=0.5,
            ),
            albu.OneOf([albu.RGBShift(), albu.HueSaturationValue()], p=0.5),
        ]
    ),
    "weak": albu.Compose([albu.HorizontalFlip()]),
    "none": albu.Compose([]),
    "geometric": albu.OneOf(
        [
            albu.HorizontalFlip(),
            albu.ShiftScaleRotate(),
            albu.Transpose(),
            albu.OpticalDistortion(),
            albu.ElasticTransform(),
        ]
    ),
}
available_size_augmentations = ["none", "resize", "random", "center"]


def get_transforms(config: Dict):
    size = config["size"]
    scope = config.get("augmentation_scope", "none")
    size_transform = config.get("size_transform", "none")

    images_normalization = config.get("images_normalization", "default")
    masks_normalization = config.get("masks_normalization", "div255")

    images_output_format_type = config.get("images_output_format_type", "float")
    masks_output_format_type = config.get("masks_output_format_type", "float")

    size_augmentations = {
        "none": albu.NoOp(),
        "resize": albu.Resize(height=size, width=size),
        "random": albu.RandomCrop(size, size),
        "center": albu.CenterCrop(size, size),
    }

    pipeline = albu.Compose(
        [
            albu.PadIfNeeded(p=1, min_height=size, min_width=size),
            augmentations[scope],
            size_augmentations[size_transform],
        ]
    )

    def process(image, mask):
        r = pipeline(image=image, mask=mask)

        transformed_image = output_format[images_output_format_type](
            normalization[images_normalization](r["image"])
        )

        transformed_mask = output_format[masks_output_format_type](
            normalization[masks_normalization](np.rint(r["mask"][:, :, 0:1]))
        )

        return transformed_image, transformed_mask

    return process
