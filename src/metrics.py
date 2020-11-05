import numpy as np
import torch
from segmentation_models_pytorch.utils import base
from segmentation_models_pytorch.utils.base import Activation


def miou(pr, gt, eps=1e-7, threshold=None, ignore_channels=None):
    """test
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: IoU (Jaccard) score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)

    ious = []
    for i in range(gt.shape[1]):
        intersection = torch.sum(gt[:, i, :, :] * pr[:, i, :, :])
        union = (
            torch.sum(gt[:, i, :, :]) + torch.sum(pr[:, i, :, :]) - intersection + eps
        )
        ious.append((intersection + eps) / union)
    return torch.mean(torch.stack(ious))


class mIoU(base.Metric):
    __name__ = "miou_score"

    def __init__(
        self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return miou(
            y_pr,
            y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )


def _take_channels(*xs, ignore_channels=None):
    if ignore_channels is None:
        return xs
    else:
        channels = [
            channel
            for channel in range(xs[0].shape[1])
            if channel not in ignore_channels
        ]
        xs = [
            torch.index_select(x, dim=1, index=torch.tensor(channels).to(x.device))
            for x in xs
        ]
        return xs


def _threshold(x, threshold=None):
    if threshold is not None:
        return (x > threshold).type(x.dtype)
    else:
        return x


def fwiou(pr, gt, eps=1e-7, frequency=None, threshold=None, ignore_channels=None):
    """Calculate Frequency Weighted IoU between ground truth and prediction
    Args:
        pr (torch.Tensor): predicted tensor
        gt (torch.Tensor):  ground truth tensor
        eps (float): epsilon to avoid zero division
        threshold: threshold for outputs binarization
    Returns:
        float: IoU (Jaccard) score
    """

    pr = _threshold(pr, threshold=threshold)
    pr, gt = _take_channels(pr, gt, ignore_channels=ignore_channels)
    if len(pr.size()) < 4:
        pr = [(pr == v) for v in range(8)]
        pr = torch.stack(pr, dim=1)
        gt = [(gt == v) for v in range(8)]
        gt = torch.stack(gt, dim=1)

    bs, c, h, w = pr.size()
    if frequency is None:
        frequency = [1] * c
    weight = np.stack([np.full((h, w), f) for f in frequency])
    weight = torch.from_numpy(weight).repeat(bs, 1, 1, 1).to("cuda")
    intersection = torch.sum(gt * pr * weight)
    union = torch.sum(gt * weight) + torch.sum(pr * weight) - intersection + eps
    return (intersection + eps) / union


class FWIoU(base.Metric):
    __name__ = "fwiou_score"

    def __init__(
        self,
        eps=1e-7,
        threshold=0.5,
        frequency=None,
        activation=None,
        ignore_channels=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels
        self.frequency = frequency

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return fwiou(
            y_pr,
            y_gt,
            frequency=self.frequency,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )