import itertools

import numpy as np
import torch


def mean(l, ignore_nan=False, empty=0):
    l = iter(l)
    if ignore_nan:
        l = itertools.ifilterfalse(np.isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == "raise":
            raise ValueError("Empty mean")
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


class ModelAdapter:
    def get_metrics(self, output=None, target=None):
        raise NotImplementedError()

    def get_loss(self, combined_loss):
        raise NotImplementedError()

    @staticmethod
    def get_input(data):
        raise NotImplementedError()

    @staticmethod
    def get_model_export(net, sample_input_shape):
        test_tensor = torch.ones(sample_input_shape).float().cpu()

        # Switches model to cpu and back to cuda for training
        traced_script = torch.jit.trace(
            net.eval().module.cpu(), test_tensor, check_trace=True
        )

        if torch.cuda.is_available():
            _ = net.module.cuda()
        else:
            _ = net.module.cpu()

        return net.module.state_dict(), traced_script


class HeatmapModelAdapter(ModelAdapter):
    def __init__(self):
        super(HeatmapModelAdapter, self).__init__()
        self.iou = 0

    def get_metrics(self, output=None, target=None):
        return {"IoU": self.iou}

    def get_loss(self, loss):
        loss_dict = {"iou_loss": loss[0].item()}
        self.iou = 1 - loss[0].item()
        return loss[0], loss_dict

    @staticmethod
    def get_input(data):

        if torch.cuda.is_available():
            inputs = data["img"].cuda()
            targets = data["heatmap"].cuda()
        else:
            inputs = data["img"].cpu()
            targets = data["heatmap"].cpu()

        return inputs, targets


class SegmentationModelAdapter(ModelAdapter):
    def __init__(self):
        super(SegmentationModelAdapter, self).__init__()
        self.iou = 0

    @staticmethod
    def _iou_binary(preds, labels, EMPTY=1.0, per_image=False):
        """
        IoU for foreground class
        binary: 1 foreground, 0 background
        """
        preds = torch.cat([p.max(dim=0)[1].byte() for p in preds])
        preds = ((preds.float().view(-1, 1, 1, 1)).data > 0).float()
        labels = labels.view(-1, 1, 1, 1)
        if not per_image:
            preds, labels = (preds,), (labels,)
        ious = []
        ious_th = []
        for pred, label in zip(preds, labels):
            intersection = ((label == 1) & (pred == 1)).sum()
            union = ((label == 1) | (pred == 1)).sum()
            if not union:
                iou = EMPTY
            else:
                iou = intersection.item() / union.item()
            thresholds = torch.arange(0.5, 1, 0.05)
            iou_th = []
            for thresh in thresholds:
                iou_th.append(iou > thresh)

            ious_th.append(np.mean(iou_th))
            ious.append(iou)

        iou = mean(ious)
        iou_th = mean(ious_th)
        return 100 * iou, 100 * iou_th

    def get_metrics(self, output=None, target=None):
        iou, thresholded_iou = self._iou_binary(output, target)
        return {"IoU": iou, "Thresholded_IoU": thresholded_iou}

    def get_loss(self, loss):
        loss_dict = {"loss": loss.item()}
        return loss, loss_dict

    @staticmethod
    def get_input(data):
        inputs = data["img"]
        targets = data["mask"]

        if torch.cuda.is_available():
            inputs, targets = (inputs.cuda(), targets.cuda())
        else:
            inputs, targets = (inputs.cpu(), targets.cpu())

        return inputs, targets


def get_model_adapter(config):
    if config["task"] == "counting":
        return HeatmapModelAdapter()
    return SegmentationModelAdapter()
