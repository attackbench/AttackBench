from functools import partial
from typing import Callable, Optional

import torch
from torch import Tensor, nn


def adv_lib_wrapper(attack: Callable,
                    model: nn.Module,
                    inputs: Tensor,
                    labels: Tensor,
                    targets: Optional[Tensor] = None,
                    targeted: bool = False) -> Tensor:
    attack_labels = targets if targeted else labels
    return attack(model=model, inputs=inputs, labels=attack_labels, targeted=targeted)


def adv_lib_minimal_wrapper(model: nn.Module,
                            inputs: Tensor,
                            labels: Tensor,
                            attack: partial,
                            init_eps: float,
                            eps_name: str = 'eps',
                            targeted: bool = False,
                            max_eps: Optional[float] = None,
                            search_steps: int = 20) -> Tensor:
    device = inputs.device
    batch_size = len(inputs)

    adv_inputs = inputs.clone()
    eps_low = torch.full((batch_size,), 0, dtype=inputs.dtype, device=device)
    best_eps = torch.full_like(eps_low, float('inf') if max_eps is None else 2 * max_eps)
    found_high = torch.full_like(eps_low, False, dtype=torch.bool)

    eps = torch.full_like(eps_low, init_eps)
    for i in range(search_steps):
        adv_inputs_run = attack(model=model, inputs=inputs, labels=labels, targeted=targeted, **{eps_name: eps})

        logits = model(adv_inputs_run)
        preds = logits.argmax(dim=1)
        is_adv = (preds == labels) if targeted else (preds != labels)

        better_adv = is_adv & (eps < best_eps)
        adv_inputs[better_adv] = adv_inputs_run[better_adv]

        found_high.logical_or_(better_adv)
        eps_low = torch.where(better_adv, eps_low, eps)
        best_eps = torch.where(better_adv, eps, best_eps)

        eps = torch.where(found_high | ((2 * eps_low) >= best_eps), (eps_low + best_eps) / 2, 2 * eps_low)

    return adv_inputs
