import warnings
from typing import Callable, Optional, Tuple

import torch
from adv_lib.utils.attack_utils import _default_metrics
from torch import Tensor, nn


class BackwardQueryCounter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.num_backwards = 0

    def __call__(self, module, grad_input: Tuple[Tensor], grad_output: Tuple[Tensor]) -> None:
        self.num_backwards += 1


class BenchModel(nn.Module):
    _start_event = torch.cuda.Event(enable_timing=True)
    _end_event = torch.cuda.Event(enable_timing=True)
    _benchmark_mode = False
    _elapsed_time = None

    def __init__(self, model: nn.Module, n_query_limit: Optional[int] = None):
        super(BenchModel, self).__init__()
        self.n_query_limit = n_query_limit

        self.num_forwards = 0
        self.backward_counter = BackwardQueryCounter()
        model.register_full_backward_hook(self.backward_counter)

        self.model = model
        self.num_classes = None

    def forward(self, input: Tensor) -> Tensor:
        if self.can_query:
            output = self.model(input)
            self.num_forwards += 1

            if self.num_classes is None:  # get number of classes from first inference
                self.num_classes = output.shape[1]

            if self._benchmark_mode:
                self.track_optimization(input=input, output=output)

        else:
            # prevents meaningful forward and backward without breaking computations graph and attack logic
            output = input.flatten(1)[:, :self.num_classes] * 0

        return output

    @property
    def num_backwards(self) -> int:
        return self.backward_counter.num_backwards

    def reset_counters(self):
        self.num_forwards = 0
        self.backward_counter.reset()

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

    def register_batch(self, inputs: Tensor, labels: Tensor, targeted: bool) -> None:
        self.inputs = inputs
        self.labels = labels
        self.batch_size = inputs.shape[0]
        self.device = inputs.device
        self.targeted = targeted

    def start_tracking(self, tracking_metric: Callable, tracking_threat_model: str) -> None:
        self.reset_counters()
        # init metrics
        self.tracking_metric = tracking_metric
        self.tracking_threat_model = tracking_threat_model
        self.metrics = _default_metrics
        self.min_dist = {m: torch.full((self.batch_size,), float('inf'), device=self.device) for m in self.metrics}

        self._benchmark_mode = True
        self._elapsed_time = None
        self.start_timing()

    def end_tracking(self) -> None:
        self.stop_timing()
        self._benchmark_mode = False
        # clean-up
        del self.inputs, self.labels, self.batch_size, self.device, self.targeted, self.min_dist, self.metrics

    def track_optimization(self, input: Tensor, output: Tensor) -> None:
        if (bs := len(input)) != self.batch_size:
            warnings.warn(f'Number of inputs ({bs}) different from tracked ({self.batch_size}) -> cannot track.')
            return

        if self.is_out_of_query_budget:
            self.stop_timing()
            print('Tracking off. Out of query budget and time already set.')
        else:
            predictions = output.argmax(dim=1)
            success = (predictions == self.labels) if self.targeted else (predictions != self.labels)

            if success.any():
                current_distances = self.tracking_metric(input[success], self.inputs[success])
                better_distance = current_distances < self.min_dist[self.tracking_threat_model][success]
                success.masked_scatter_(success, better_distance)  # cannot use [] indexing with self
                if success.any():
                    modified_inputs, original_inputs = input[success], self.inputs[success]
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
