from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.autograd import Variable


class IoULoss(nn.Module):
    """ IoU loss
    """

    def __init__(self):
        super(IoULoss, self).__init__()

    @staticmethod
    def iou_metric(y_pred: Tensor, y_true: Tensor):
        # small value to prevent division by zero
        _EPSILON = np.finfo(np.float32).eps

        op_sum = lambda x: x.sum(2).sum(2)

        intersection = op_sum(y_true * y_pred) + _EPSILON

        union = (
            op_sum(y_true ** 2)
            + op_sum(y_pred ** 2)
            - op_sum(y_true * y_pred)
            + _EPSILON
        )

        loss = intersection / union

        loss = torch.mean(loss)
        return loss

    def forward(self, y_pred, y_true):
        """ Compute IoU loss
        Args:
            y_pred (torch.Tensor): predicted values
            y_true (torch.Tensor): target values
        """
        return 1 - self.iou_metric(torch.sigmoid(y_pred), y_true)


class FocalLoss(nn.Module):
    def __init__(self, gamma):
        super().__init__()
        self.gamma = gamma

    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError(
                "Target size ({}) must be the same as input size ({})".format(
                    target.size(), input.size()
                )
            )
        max_val = (-input).clamp(min=0)
        loss = (
            input
            - input * target
            + max_val
            + ((-max_val).exp() + (-input - max_val).exp()).log()
        )
        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        return loss.mean()


class LovaszLoss(nn.Module):
    def __init__(self):
        super(LovaszLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    @staticmethod
    def _lovasz_grad(gt_sorted):
        """
        Computes gradient of the Lovasz extension w.r.t sorted errors
        See Alg. 1 in paper
        """
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1.0 - intersection / union
        if p > 1:  # cover 1-pixel case
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        return jaccard

    @staticmethod
    def _flatten_binary_scores(scores, labels, ignore=None):
        """
        Flattens predictions in the batch (binary case)
        Remove labels equal to 'ignore'
        """
        scores = scores.view(-1)
        labels = labels.view(-1)
        if ignore is None:
            return scores, labels
        valid = labels != ignore
        vscores = scores[valid]
        vlabels = labels[valid]
        return vscores, vlabels

    def _lovasz_hinge_flat(self, logits, labels):
        """
        Binary Lovasz hinge loss
          logits: [P] Variable, logits at each prediction (between -\infty and +\infty)
          labels: [P] Tensor, binary ground truth labels (0 or 1)
        """
        if len(labels) == 0:
            return logits.sum() * 0.0
        signs = 2.0 * labels.float() - 1.0
        errors = 1.0 - logits * Variable(signs)
        errors_sorted, perm = torch.sort(errors, dim=0, descending=True)
        perm = perm.data
        gt_sorted = labels[perm]
        grad = self._lovasz_grad(gt_sorted)
        loss = torch.dot(F.elu(errors_sorted) + 1, Variable(grad))
        return loss

    def forward(self, y_pred, y_true):
        """
        Binary Lovasz hinge loss
            y_pred: [B, H, W] Variable, logits at each pixel (between -\infty and +\infty)
            y_true: [B, H, W] Tensor, binary ground truth masks (0 or 1)
        """
        return self._lovasz_hinge_flat(*self._flatten_binary_scores(y_pred, y_true))


class MixedLoss(nn.Module):
    def __init__(self, alpha=10.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.focal = FocalLoss(gamma)
        self.dice = DiceLoss()

    def forward(self, input: Tensor, target: Tensor):
        loss = self.alpha * self.focal(input, target) - torch.log(
            self.dice(input, target)
        )
        return loss.mean()


class Espnet2CrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()
        if torch.cuda.is_available():
            self.crossentropy = nn.CrossEntropyLoss(
                torch.FloatTensor([0.1, 0.9]).cuda()
            )
        else:
            self.crossentropy = nn.CrossEntropyLoss(torch.FloatTensor([0.1, 0.9]).cpu())

    def forward(self, input, target):
        if isinstance(input, Tuple) and len(input) == 2:
            input1, input2 = input
            batch_size, _, w, h = target.shape
            target = target.view(batch_size, w, h)
            return self.crossentropy(input1, target) + self.crossentropy(input2, target)
        else:
            batch_size, _, w, h = target.shape
            target = target.view(batch_size, w, h)
            return self.crossentropy(input, target)


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()

    @staticmethod
    def dice_loss(input: Tensor, target: Tensor):
        input = torch.sigmoid(input)
        smooth = 1.0
        iflat = input.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()
        return (2.0 * intersection + smooth) / (iflat.sum() + tflat.sum() + smooth)

    def forward(self, input: Tensor, target: Tensor):
        return self.dice_loss(input, target)


losses = {
    "l2_loss": nn.MSELoss,
    "iou_loss": IoULoss,
    "dice_loss": DiceLoss,
    "cross_entropy": nn.CrossEntropyLoss,
    "bce": nn.BCEWithLogitsLoss,
    "lovasz": LovaszLoss,
    "focal_loss": FocalLoss,
    "mixed_loss": MixedLoss,
    "espnet_crossentropy_loss": Espnet2CrossEntropyLoss,
}


def get_loss(loss):
    """ Creates loss from config
    Args:
        loss (dict): dictionary of loss configuration:
        - name (str): loss name
        and other configs for specified loss
    """
    loss_name = loss["name"]

    try:
        return losses[loss_name]()
    except KeyError:
        raise ValueError("Loss [%s] not recognized." % loss_name)
