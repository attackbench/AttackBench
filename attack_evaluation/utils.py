import warnings
from distutils.version import LooseVersion
from typing import Callable, Dict, Optional, Union

import torch
from adv_lib.utils import BackwardCounter, ForwardCounter
from adv_lib.utils.attack_utils import _default_metrics
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def run_attack(model: nn.Module,
               loader: DataLoader,
               attack: Callable,
               targets: Optional[Union[int, Tensor]] = None,
               metrics: Dict[str, Callable] = _default_metrics,
               return_adv: bool = False) -> dict:
    device = next(model.parameters()).device
    targeted = True if targets is not None else False
    loader_length = len(loader)

    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    forward_counter, backward_counter = ForwardCounter(), BackwardCounter()
    model.register_forward_pre_hook(forward_counter)
    if LooseVersion(torch.__version__) >= LooseVersion('1.8'):
        model.register_full_backward_hook(backward_counter)
    else:
        model.register_backward_hook(backward_counter)
    forwards, backwards = [], []  # number of forward and backward calls per sample

    times, accuracies, ori_success, adv_success = [], [], [], []
    distances = {k: [] for k in metrics.keys()}

    if return_adv:
        all_inputs, all_adv_inputs = [], []

    for inputs, labels in tqdm(loader, ncols=80, total=loader_length):
        if return_adv:
            all_inputs.append(inputs.clone())

        inputs, labels = inputs.to(device), labels.to(device)

        logits = model(inputs)
        predictions = logits.argmax(dim=1)
        accuracies.extend((predictions == labels).cpu().tolist())
        success = (predictions == targets) if targeted else (predictions != labels)
        ori_success.extend(success.cpu().tolist())

        forward_counter.reset(), backward_counter.reset()
        start.record()
        adv_inputs = attack(model=model, inputs=inputs, labels=labels, targeted=targeted, targets=targets)
        # performance monitoring
        end.record()
        torch.cuda.synchronize()
        times.append((start.elapsed_time(end)) / 1000)  # times for cuda Events are in milliseconds
        forwards.append(forward_counter.num_samples_called)
        backwards.append(backward_counter.num_samples_called)
        forward_counter.reset(), backward_counter.reset()

        if adv_inputs.min() < 0 or adv_inputs.max() > 1:
            warnings.warn('Values of produced adversarials are not in the [0, 1] range -> Clipping to [0, 1].')
            adv_inputs.clamp_(min=0, max=1)

        if return_adv:
            all_adv_inputs.append(adv_inputs.cpu().clone())

        adv_logits = model(adv_inputs)
        adv_pred = adv_logits.argmax(dim=1)

        success = (adv_pred == targets) if targeted else (adv_pred != labels)
        adv_success.extend(success.cpu().tolist())

        for metric, metric_func in metrics.items():
            distances[metric].extend(metric_func(adv_inputs, inputs).detach().cpu().tolist())

    data = {
        'targeted': targeted,
        'accuracy': sum(accuracies) / len(accuracies),
        'ori_success': ori_success,
        'adv_success': adv_success,
        'ASR': sum(adv_success) / len(adv_success),
        'times': times,
        'num_forwards': forwards,
        'num_backwards': backwards,
        'distances': distances,
    }

    if return_adv:
        shapes = [img.shape for img in inputs]
        if len(set(shapes)) == 1:
            inputs = torch.cat(inputs, dim=0)
            adv_inputs = torch.cat(adv_inputs, dim=0)
        data['inputs'] = inputs
        data['adv_inputs'] = adv_inputs

    return data
