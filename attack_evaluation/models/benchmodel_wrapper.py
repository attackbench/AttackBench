from torch import nn
from distutils.version import LooseVersion
import torch


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

    def reset_query_budget(self):
        self.forward_counter.reset()
        self.backward_counter.reset()
        self.query_stopper.reset(self.forward_counter, self.backward_counter)

    def forward(self, x):
        """
        if self.is_out_of_query_budget():
            with torch.no_grad():
                print(x.shape, (x - self.inputs).norm(2))
            return torch.zeros_like(self.model(x))
        """
        print('#forwards: ', self.forward_counter.num_queries_called)
        print('#backwards: ', self.backward_counter.num_queries_called)
        #with torch.no_grad():
        #    print(x.shape, (x-self.inputs).norm(2))
        return self.model(x)


    def register_batch(self, inputs):
        self.inputs = inputs

    """
    def is_out_of_query_budget(self):
        if self.n_query_limit is None:
            return None

        n_forwards = self.forward_counter.num_queries_called
        n_backwards = self.backward_counter.num_queries_called
        return (n_forwards + n_backwards) >= self.n_query_limit
    """