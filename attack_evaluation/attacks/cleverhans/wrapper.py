import inspect
from typing import Callable, Optional

import torch
from torch import Tensor, nn


def cleverhans_wrapper(attack: Callable,
                       model: nn.Module,
                       inputs: Tensor,
                       labels: Tensor,
                       targets: Optional[Tensor] = None,
                       targeted: bool = False) -> Tensor:
    if attack.func == cleverhans_minimal_wrapper:
        cleverhans_attack = attack.keywords['attack'].func
    else:
        cleverhans_attack = attack.func
    attack_parameters = inspect.signature(cleverhans_attack).parameters
    attack_kwargs = {}
    if 'n_classes' in attack_parameters:  # specify the number of classes for some attacks
        attack_kwargs['n_classes'] = [module for module in model.modules()][-1].out_features
    if 'targeted' in attack_parameters:
        attack_kwargs['targeted'] = targeted
    if 'y' in attack_parameters:
        attack_kwargs['y'] = targets if targeted else labels
    if 'y_target' in attack_parameters and targeted:
        attack_kwargs['y_target'] = targets

    adv_examples = attack(model_fn=model, x=inputs, **attack_kwargs)
    return adv_examples


def cleverhans_minimal_wrapper(attack: Callable,
                               model_fn: nn.Module,
                               x: Tensor,
                               init_eps: float,
                               max_eps: Optional[float] = None,
                               search_steps: int = 20, **kwargs) -> Tensor:
    device = x.device
    batch_size = len(x)

    labels = kwargs.get('y', model_fn(x).argmax(dim=1))
    targeted = kwargs.get('targeted', False)

    adv_inputs = x.clone()
    eps_low = torch.full((batch_size,), 0, dtype=x.dtype, device=device)
    best_eps = torch.full_like(eps_low, float('inf') if max_eps is None else 2 * max_eps)
    found_high = torch.full_like(eps_low, False, dtype=torch.bool)

    eps = torch.full_like(eps_low, init_eps)
    for i in range(search_steps):
        adv_inputs_run = x.clone()
        for eps_ in torch.unique(eps):
            mask = eps == eps_
            eps_kwargs = {k: v for k, v in kwargs.items() if k in ['n_classes', 'targeted']}
            if 'y' in kwargs:
                eps_kwargs['y'] = kwargs['y'][mask]
            if 'y_target' in kwargs:
                eps_kwargs['y_target'] = kwargs['y_target'][mask]
            adv_inputs_run[mask] = attack(model_fn=model_fn, x=x[mask], eps=eps_, **eps_kwargs)

        logits = model_fn(adv_inputs_run)
        preds = logits.argmax(dim=1)
        is_adv = (preds == labels) if targeted else (preds != labels)

        better_adv = is_adv & (eps < best_eps)
        adv_inputs[better_adv] = adv_inputs_run[better_adv]

        found_high.logical_or_(better_adv)
        eps_low = torch.where(better_adv, eps_low, eps)
        best_eps = torch.where(better_adv, eps, best_eps)

        eps = torch.where(found_high | ((2 * eps_low) >= best_eps), (eps_low + best_eps) / 2, 2 * eps_low)

    return adv_inputs
