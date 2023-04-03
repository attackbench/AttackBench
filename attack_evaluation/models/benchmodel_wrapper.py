from torch import nn
from distutils.version import LooseVersion
import torch
from adv_lib.utils.attack_utils import _default_metrics


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
        self.threat_model = 2  # TODO: adapt to the scenario
        self.benchmark_mode = False

    def forward(self, x):
        """
        if self.is_out_of_query_budget():
            with torch.no_grad():
                print(x.shape, (x - self.inputs).norm(2))
            return torch.zeros_like(self.model(x))
        """
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
        self.x_origin = inputs
        self.y_origin = self._predict_no_forward_counting(inputs)

    def init_metrics(self):
        self.metrics = _default_metrics
        self.min_dist = {k: [] for k in self.metrics.keys()}

    def start_tracking(self, inputs):
        self.reset_query_budget()
        self.register_batch(inputs)
        self.init_metrics()
        self.benchmark_mode = True

    def end_tracking(self):
        del self.x_origin
        del self.y_true
        self.benchmark_mode = False

    @torch.no_grad()
    def track_optimization(self, x):
        if not self.is_out_of_query_budget():
            predictions = self._predict_no_forward_counting(x)
            success = (predictions != self.y_origin)  # TODO: need to adapt in case of targeted attacks

            if success.any():
                idx_success = torch.nonzero(success, as_tuple=True)
                current_delta = (x[idx_success] - self.x_origin[idx_success])
                current_distances = current_delta.flatten(1).norm(self.threat_model, dim=1)

                threat_model_str = 'l' + str(self.threat_model)

                if not len(self.min_dist[threat_model_str]) > 0:
                    # init best distances to float inf.
                    # we init a vector with best distances equal to infinity
                    self.min_dist[threat_model_str] = (
                                torch.ones(x.shape[0], requires_grad=False) * float('inf')).to(x.device)

                dist_success = current_distances < self.min_dist[threat_model_str][idx_success]
                if dist_success.any():
                    dist_success_idx = torch.nonzero(dist_success, as_tuple=True)
                    distances_to_change = idx_success[0][dist_success_idx]
                    self.min_dist[threat_model_str][distances_to_change] = current_distances[dist_success]

                    #for metric, metric_func in self.metrics.items():
                    #    metric_str = 'l'+str(metric)
                    #    self.min_dist[metric_str][distances_to_change] = metric_func(x, self.x_origin).detach().cpu().tolist()

    @torch.no_grad()
    def _predict_no_forward_counting(self, x):
        logits = self.model(x)
        predictions = logits.argmax(dim=1)
        self.forward_counter.num_queries_called -= 1
        return predictions
