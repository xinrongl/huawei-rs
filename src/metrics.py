from src.utils import base
from src.utils import functional as F
from src.utils.base import Activation


class IoU(base.Metric):
    __name__ = "iou_score"

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
        return F.iou(
            y_pr,
            y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )


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
        return F.miou(
            y_pr,
            y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )


class cemIoU(base.Metric):
    __name__ = "cemiou"

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
        return F.cemiou(
            y_pr,
            y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )


class classIoU1(base.Metric):
    __name__ = "class_iou1"

    def __init__(
        self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels
        self.class_idx = 0

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.class_iou(
            y_pr,
            y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
            class_idx=self.class_idx,
        )


class classIoU2(base.Metric):
    __name__ = "class_iou2"

    def __init__(
        self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels
        self.class_idx = 1

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.class_iou(
            y_pr,
            y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
            class_idx=self.class_idx,
        )


class ceIoU1(base.Metric):
    __name__ = "ce_iou1"

    def __init__(
        self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels
        self.class_idx = 0

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.ce_iou(
            y_pr,
            y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
            class_idx=self.class_idx,
        )


class ceIoU2(base.Metric):
    __name__ = "ce_iou2"

    def __init__(
        self, eps=1e-7, threshold=0.5, activation=None, ignore_channels=None, **kwargs
    ):
        super().__init__(**kwargs)
        self.eps = eps
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels
        self.class_idx = 1

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.ce_iou(
            y_pr,
            y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
            class_idx=self.class_idx,
        )


class Fscore(base.Metric):
    def __init__(
        self,
        beta=1,
        eps=1e-7,
        threshold=0.5,
        activation=None,
        ignore_channels=None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.f_score(
            y_pr,
            y_gt,
            eps=self.eps,
            beta=self.beta,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )


class Accuracy(base.Metric):
    def __init__(self, threshold=0.5, activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.threshold = threshold
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return F.accuracy(
            y_pr,
            y_gt,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )


class Recall(base.Metric):
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
        return F.recall(
            y_pr,
            y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )


class Precision(base.Metric):
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
        return F.precision(
            y_pr,
            y_gt,
            eps=self.eps,
            threshold=self.threshold,
            ignore_channels=self.ignore_channels,
        )
