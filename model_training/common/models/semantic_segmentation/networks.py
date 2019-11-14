import functools
from typing import Dict

import torch
import torch.nn as nn

from model_training.common.models.semantic_segmentation.espnet import EESPNet_Seg
from model_training.common.models.semantic_segmentation.fpn_inception import (
    FPNInception,
)
from model_training.common.models.semantic_segmentation.fpn_mobilenet import (
    FPNMobileNet,
)
from model_training.common.models.semantic_segmentation.unet_densenet import (
    UNetDenseNet,
)
from model_training.common.models.semantic_segmentation.unet_resnet import UNetResNet
from model_training.common.models.semantic_segmentation.unet_seresnext import (
    UNetSEResNext,
)

available_architectures = [
    "fpn_mobilenet",
    "fpn_inception",
    "unet_seresnext",
    "unet_densenet",
    "unet_resnet",
    "espnet",
]


def get_net(model_config: Dict) -> nn.DataParallel:
    """
    Factory method for getting model depending on a given config.
    :param model_config:
    :raise NotImplementedError: if network architecture name not recognized.
    :return: model
    """
    networks = {
        "fpn_mobilenet": lambda: FPNMobileNet(
            pretrained=model_config["pretrained"], num_classes=model_config["classes"]
        ),
        "fpn_inception": lambda: FPNInception(
            norm_layer=_get_norm_layer(),
            pretrained=model_config["pretrained"],
            num_classes=model_config["classes"],
        ),
        "unet_seresnext": lambda: UNetSEResNext(
            pretrained=model_config["pretrained"],
            num_classes=model_config["classes"],
            encoder_depth=model_config["depth"],
            norm_layer=_get_norm_layer(),
        ),
        "unet_densenet": lambda: UNetDenseNet(
            pretrained=model_config["pretrained"],
            num_classes=model_config["classes"],
            encoder_depth=model_config["depth"],
        ),
        "unet_resnet": lambda: UNetResNet(
            pretrained=model_config["pretrained"],
            num_classes=model_config["classes"],
            encoder_depth=model_config["depth"],
        ),
        "espnet": lambda: _get_espnet(model_config),
    }

    arch_name = model_config.get("arch", "unknown")
    try:
        net = networks[arch_name]()
    except KeyError:
        raise ValueError("Network [%s] not recognized." % arch_name)
    else:
        return nn.DataParallel(net)


def _get_norm_layer(norm_type: str = "batch") -> object:
    norm_layer = {
        "batch": lambda: functools.partial(nn.BatchNorm2d, affine=True),
        "instance": lambda: functools.partial(nn.InstanceNorm2d, affine=False),
    }

    try:
        return norm_layer[norm_type]()
    except KeyError:
        raise NotImplementedError("normalization layer [%s] is not found" % norm_type)


def _get_espnet(model_config: Dict) -> EESPNet_Seg:
    """
    Load espnet model depending on a given model config.
    :param model_config:
    :return: EESPNet_Seg model
    """
    if model_config["pretrained"]:
        pretrained_model = torch.load(model_config["pretrained_weights_path"])
        if "model" in pretrained_model.keys():
            net = EESPNet_Seg(pretrained=None, classes=model_config["classes"])
            net.load_state_dict(pretrained_model["model"])
        else:
            net = EESPNet_Seg(
                pretrained=model_config["pretrained_weights_path"],
                classes=model_config["classes"],
            )
    else:
        net = EESPNet_Seg(pretrained=None, classes=model_config["classes"])

    return net
