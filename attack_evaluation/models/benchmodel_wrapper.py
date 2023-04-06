import warnings
from typing import Callable, Optional, Tuple

import torch
from adv_lib.utils.attack_utils import _default_metrics
from torch import Tensor, nn
from torch.nn import functional as F


class BenchModel(nn.Module):
    def __init__(self, model: nn.Module, enforce_box: bool = True, n_query_limit: Optional[int] = None):
        super(BenchModel, self).__init__()
        self.enforce_box = enforce_box
        self.n_query_limit = n_query_limit

        self.model = model
        model.register_full_backward_hook(self.backward_hook)
        self.num_classes = None
        self._benchmark_mode = False

    def forward(self, input: Tensor) -> Tensor:
        if self.can_query:
            if self.enforce_box:
                input = input.clamp(min=0, max=1)

            output = self.model(input)

            if self._benchmark_mode:
                self.num_forwards += 1
                self.track_optimization(input=input, output=output)

        else:
            # prevents meaningful forward and backward without breaking computations graph and attack logic
            sign = -1 if self.targeted else 1
            one_hot = F.one_hot(self.targets if self.targeted else self.labels, num_classes=self.num_classes)
            output = input.flatten(1).narrow(1, 0, 1) * 0 + sign * one_hot

        return output

    def backward_hook(self, module, grad_input: Tuple[Tensor], grad_output: Tuple[Tensor]) -> Optional[Tuple[Tensor]]:
        if self.can_query:
            if self._benchmark_mode:
                self.num_backwards += 1
        else:
            return tuple(torch.zeros_like(g) for g in grad_input)

    def reset_counters(self):
        self.num_forwards = 0
        self.num_backwards = 0

    @property
    def is_out_of_query_budget(self) -> bool:
        if self.n_query_limit is None:
            return False

        return (self.num_forwards + self.num_backwards) >= self.n_query_limit

    @property
    def can_query(self) -> bool:
        if self._benchmark_mode is False or self.n_query_limit is None:
            return True

        return (self.num_forwards + self.num_backwards) < self.n_query_limit

    def start_tracking(self, inputs: Tensor, labels: Tensor, targeted: bool, tracking_metric: Callable,
                       tracking_threat_model: str, targets: Optional[Tensor] = None) -> None:
        self.inputs = inputs
        self.labels = labels
        self.batch_size = len(inputs)
        self.device = inputs.device
        self.targeted = targeted
        self.targets = targets

        # check if inputs are already adversarial (i.e. misclassified for untargeted)
        logits = self.forward(inputs)
        self.num_classes = logits.shape[1]
        predictions = logits.argmax(dim=1)
        self.correct = predictions == labels
        self.ori_success = (predictions == labels) if targeted else (predictions != labels)

        # init metrics
        self.reset_counters()
        self.tracking_metric = tracking_metric
        self.tracking_threat_model = tracking_threat_model
        self.metrics = _default_metrics
        self.min_dist = {m: torch.full((self.batch_size,), float('inf'), device=self.device) for m in self.metrics}
        for dists in self.min_dist.values():
            dists.masked_fill_(self.ori_success, 0)  # replace minimum distances with 0 for already adversarial inputs

        # timing objects
        self._start_event = torch.cuda.Event(enable_timing=True)
        self._end_event = torch.cuda.Event(enable_timing=True)
        self._benchmark_mode = True
        self._elapsed_time = None
        torch.cuda.synchronize()
        self.start_timing()

    def end_tracking(self) -> None:
        self.stop_timing()
        self._benchmark_mode = False

    def track_optimization(self, input: Tensor, output: Tensor) -> None:
        if (bs := len(input)) != self.batch_size:
            warnings.warn(f'Number of inputs ({bs}) different from tracked ({self.batch_size}) -> cannot track.')
            return

        if self.is_out_of_query_budget:
            self.stop_timing()
            print('Tracking off. Out of query budget and time already set.')
        else:
            predictions = output.argmax(dim=1)
            success = (predictions == self.targets) if self.targeted else (predictions != self.labels)

            if success.any():
                current_distances = self.tracking_metric(input.detach()[success], self.inputs[success])
                better_distance = current_distances < self.min_dist[self.tracking_threat_model][success]
                success.masked_scatter_(success, better_distance)  # cannot use [] indexing with self
                if success.any():
                    modified_inputs, original_inputs = input.detach()[success], self.inputs[success]
                    for metric, metric_func in self.metrics.items():
                        self.min_dist[metric][success] = metric_func(modified_inputs, original_inputs, dim=1)

    def start_timing(self) -> None:
        self._start_event.record()

    def stop_timing(self) -> None:
        self._end_event.record()
        torch.cuda.synchronize()
        self._elapsed_time = self._start_event.elapsed_time(
            self._end_event) / 1000  # times for cuda Events are in milliseconds

    @property
    def elapsed_time(self) -> float:
        if self._elapsed_time is None:
            self._end_event.record()
            torch.cuda.synchronize()
            return self._start_event.elapsed_time(self._end_event) / 1000
        else:
            return self._elapsed_time
