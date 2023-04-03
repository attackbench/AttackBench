from torch import nn
from distutils.version import LooseVersion
import torch
from adv_lib.utils.attack_utils import _default_metrics
from typing import Callable, Dict, Optional, Union


class ForwardQueryCounter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.num_queries_called = 0

    def __call__(self, module, input) -> None:
        self.num_queries_called += 1


class BackwardQueryCounter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.num_queries_called = 0

    def __call__(self, module, grad_input, grad_output) -> None:
        self.num_queries_called += 1


class StopBackwardQuery:
    def __init__(self, forward_counter, backward_counter, n_query_limit):
        self.reset(forward_counter, backward_counter)
        self.n_query_limit = n_query_limit

    def reset(self, forward_counter, backward_counter):
        self.forward_counter = forward_counter
        self.backward_counter = backward_counter

    def __call__(self, module, grad_input, grad_output) -> None:
        if self.is_out_of_query_budget():
            print('\n', "Out of query budget", '\n')
            input_gradients = (torch.zeros_like(grad_input[0]),)
            return input_gradients
        return grad_input

    def is_out_of_query_budget(self):
        if self.n_query_limit is None:
            return None

        n_forwards = self.forward_counter.num_queries_called
        n_backwards = self.backward_counter.num_queries_called
        return (n_forwards + n_backwards) > self.n_query_limit


class BenchModel(nn.Module):

    def __init__(self, model: nn.Module, n_query_limit: int):
        super(BenchModel, self).__init__()
        self.n_query_limit = n_query_limit

        self.forward_counter, self.backward_counter = ForwardQueryCounter(), BackwardQueryCounter()
        self.query_stopper = StopBackwardQuery(self.forward_counter, self.backward_counter, self.n_query_limit)

        model.register_forward_pre_hook(self.forward_counter)
        if LooseVersion(torch.__version__) >= LooseVersion('1.8'):
            model.register_full_backward_hook(self.backward_counter)
            model.register_full_backward_hook(self.query_stopper)
        else:
            model.register_backward_hook(self.backward_counter)
            model.register_backward_hook(self.query_stopper)
        self.model = model
        self.execution_time = None
        self.benchmark_mode = False

    def forward(self, x):
        if self.benchmark_mode:
            self.track_optimization(x)
            print('#forwards: ', self.forward_counter.num_queries_called)
            print('#backwards: ', self.backward_counter.num_queries_called)
        # with torch.no_grad():
        #    print(x.shape, (x-self.inputs).norm(2))
        return self.model(x)

    def is_out_of_query_budget(self):
        return self.query_stopper.is_out_of_query_budget()

    def reset_query_budget(self):
        self.forward_counter.reset()
        self.backward_counter.reset()
        self.query_stopper.reset(self.forward_counter, self.backward_counter)

    def register_batch(self, inputs):
        self.batch_size = inputs.shape[0]
        self.device = inputs.device

        self.x_origin = inputs
        self.y_origin = self._predict_no_forward_counting(inputs)

    def init_metrics(self):
        self.metrics = _default_metrics
        self.min_dist = {k: [] for k in self.metrics.keys()}

        for metric, metric_func in self.metrics.items():
            self.min_dist[metric] = torch.full((self.batch_size,), float('inf')).to(self.device)

    def start_tracking(self, inputs, tracking_metric: Callable, tracking_threat_model: str):
        self.reset_query_budget()
        self.register_batch(inputs)
        self.init_metrics()
        self.benchmark_mode = True
        self.tracking_metric = tracking_metric
        self.tracking_threat_model = tracking_threat_model

    def end_tracking(self):
        del self.x_origin, self.y_true, self.min_dist, self.metric
        self.benchmark_mode = False

    @torch.no_grad()
    def track_optimization(self, x):
        if not self.is_out_of_query_budget():
            predictions = self._predict_no_forward_counting(x)
            success = (predictions != self.y_origin)  # TODO: need to adapt in case of targeted attacks

            if success.any():
                current_distances = self.tracking_metric(x[success], self.x_origin[success])
                dist_success = current_distances < self.min_dist[self.tracking_threat_model][success]
                if dist_success.any():
                    v = torch.zeros_like(success)
                    v[success] = dist_success
                    for metric, metric_func in self.metrics.items():
                        self.min_dist[metric][v] = metric_func(x[v], self.x_origin[v], dim=1)

    @torch.no_grad()
    def _predict_no_forward_counting(self, x):
        logits = self.model(x)
        predictions = logits.argmax(dim=1)
        self.forward_counter.num_queries_called -= 1
        return predictions
