import sys

import torch
import torch.nn.functional as F
from segmentation_models_pytorch.utils.meter import AverageValueMeter
from tqdm import tqdm


class Epoch:
    def __init__(
        self,
        model,
        model_t,
        loss,
        loss2,
        metrics,
        stage_name,
        device="cpu",
        verbose=True,
    ):
        """Train model with distillation

        Args:
            model (smp model): [student model]
            model_t (smp model): [teacher model]
            loss ([type]): [student loss]
            loss2 ([type]): [teacher loss]
            metrics ([type]): [description]
            stage_name ([type]): [description]
            device (str, optional): Defaults to "cpu".
            verbose (bool, optional): Defaults to True.
        """

        self.model = model
        self.model_t = model_t
        self.loss = loss
        self.loss2 = loss2
        self.metrics = metrics
        self.stage_name = stage_name
        self.verbose = verbose
        self.device = device

        self._to_device()

    def _to_device(self):
        self.model.to(self.device)
        self.model_t.to(self.device)
        self.loss.to(self.device)
        self.loss2.to(self.device)
        for metric in self.metrics:
            metric.to(self.device)

    def _format_logs(self, logs):
        str_logs = ["{} - {:.4}".format(k, v) for k, v in logs.items()]
        s = ", ".join(str_logs)
        return s

    def batch_update(self, x, y):
        raise NotImplementedError

    def on_epoch_start(self):
        pass

    def run(self, dataloader):

        self.on_epoch_start()

        logs = {}
        loss_meter = AverageValueMeter()
        metrics_meters = {
            metric.__name__: AverageValueMeter() for metric in self.metrics
        }

        with tqdm(
            dataloader,
            desc=self.stage_name,
            file=sys.stdout,
            disable=not (self.verbose),
        ) as iterator:
            for x, y in iterator:
                x, y = x.to(self.device), y.to(self.device)
                loss, y_pred = self.batch_update(x, y)
                # update loss logs
                if self.loss.__name__ == "cross_entropy_loss":
                    _, y_pred = torch.max(y_pred, dim=1)
                loss_value = loss.cpu().detach().numpy()
                loss_meter.add(loss_value)
                loss_logs = {self.loss.__name__: loss_meter.mean}
                logs.update(loss_logs)

                # update metrics logs
                for metric_fn in self.metrics:
                    metric_value = metric_fn(y_pred, y).cpu().detach().numpy()
                    metrics_meters[metric_fn.__name__].add(metric_value)
                metrics_logs = {k: v.mean for k, v in metrics_meters.items()}
                logs.update(metrics_logs)

                if self.verbose:
                    s = self._format_logs(logs)
                    iterator.set_postfix_str(s)

        return logs


class TrainDistillEpoch(Epoch):
    def __init__(
        self,
        model,
        model_t,
        loss,
        loss2,
        metrics,
        optimizer,
        device="cpu",
        verbose=True,
    ):
        super().__init__(
            model=model,
            model_t=model_t,
            loss=loss,
            loss2=loss2,
            metrics=metrics,
            stage_name="train",
            device=device,
            verbose=verbose,
        )
        self.optimizer = optimizer

    def on_epoch_start(self):
        self.model.train()
        self.model_t.train()

    def batch_update(self, x, y):
        ALPHA = 0.5
        T = 2

        self.optimizer.zero_grad()
        prediction_s, _ = self.model.forward(x)
        loss1 = self.loss(prediction_s, y)

        prediction_t, _ = self.model_t.forward(x)

        outputs_s = F.softmax(prediction_s / T, dim=1)
        outputs_t = F.softmax(prediction_t / T, dim=1)
        loss2 = self.loss2(outputs_s, outputs_t) * T * T

        loss = loss1 * (1 - ALPHA) + loss2 * ALPHA

        loss.backward()
        self.optimizer.step()
        return loss, prediction_s


class ValidEpoch(Epoch):
    def __init__(self, model, loss, metrics, device="cpu", verbose=True):
        super().__init__(
            model=model,
            loss=loss,
            metrics=metrics,
            stage_name="valid",
            device=device,
            verbose=verbose,
        )

    def on_epoch_start(self):
        self.model.eval()

    def batch_update(self, x, y):
        with torch.no_grad():
            prediction = self.model.forward(x)
            loss = self.loss(prediction, y)
        return loss, prediction
