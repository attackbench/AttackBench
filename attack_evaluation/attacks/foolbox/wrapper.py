from functools import partial
from typing import Optional, Union

import torch
from foolbox import Misclassification, PyTorchModel, TargetedMisclassification
from foolbox.attacks.base import FixedEpsilonAttack
from torch import Tensor, nn


def foolbox_wrapper(attack: partial,
                    model: nn.Module,
                    inputs: Tensor,
                    labels: Tensor,
                    targets: Optional[Tensor] = None,
                    targeted: bool = False) -> Tensor:
    # check if attack is distance constrained, and extract epsilon
    epsilon_kwarg = {}
    attack_kwargs = attack.keywords.copy()
    if issubclass(attack.func, FixedEpsilonAttack):
        epsilon_kwarg['epsilon'] = attack_kwargs.pop('epsilon')

    fb_model = PyTorchModel(model=model, bounds=(0, 1))
    fb_attack = partial(attack.func, **attack_kwargs)()

    if hasattr(fb_attack, 'feed'):
        fb_attack.feed(model=fb_model, inputs=inputs)

    if targeted:
        criterion = TargetedMisclassification(targets)
    else:
        criterion = Misclassification(labels)

    adv_inputs = fb_attack.run(model=fb_model, inputs=inputs, criterion=criterion, **epsilon_kwarg)
    return adv_inputs


class FoolboxMinimalWrapper:
    _label_args = {
        Misclassification: 'labels',
        TargetedMisclassification: 'target_classes',
    }

    def __init__(self, attack: partial, init_eps: float, search_steps: int, max_eps: Optional[float] = None):
        self.attack = attack()
        self.init_eps = init_eps
        self.search_steps = search_steps
        self.max_eps = max_eps

    def run(self, model: PyTorchModel, inputs: Tensor,
            criterion: Union[TargetedMisclassification, Misclassification]) -> Tensor:

        # fetch labels / targets from criterion
        crit_class = criterion.__class__
        label_arg = self._label_args[crit_class]
        labels = getattr(criterion, label_arg)

        adv_inputs = inputs.clone()
        eps_low = inputs.new_zeros(len(inputs))
        best_eps = torch.full_like(eps_low, float('inf') if self.max_eps is None else 2 * self.max_eps)
        found_high = torch.full_like(eps_low, False, dtype=torch.bool)

        eps = torch.full_like(eps_low, self.init_eps)
        for i in range(self.search_steps):
            adv_inputs_run = inputs.clone()
            for eps_ in torch.unique(eps):
                mask = eps == eps_
                crit_eps = crit_class(labels[mask])
                adv_inputs_run[mask] = self.attack.run(model=model, inputs=inputs[mask], criterion=crit_eps,
                                                       epsilon=eps_.item())

            logits = model(adv_inputs_run)
            is_adv = criterion(adv_inputs_run, logits)

            better_adv = is_adv & (eps < best_eps)
            adv_inputs[better_adv] = adv_inputs_run[better_adv]

            found_high.logical_or_(better_adv)
            eps_low = torch.where(better_adv, eps_low, eps)
            best_eps = torch.where(better_adv, eps, best_eps)

            eps = torch.where(found_high | ((2 * eps_low) >= best_eps), (eps_low + best_eps) / 2, 2 * eps_low)

        return adv_inputs
