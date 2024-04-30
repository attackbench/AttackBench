import inspect
from typing import Optional

import torch
from deeprobust.image.attack.base_attack import BaseAttack
from torch import Tensor, nn


class DeepRobustModel(nn.Module):
    def __init__(self, model: nn.Module):
        super(DeepRobustModel, self).__init__()
        self.model = model

    def forward(self, x: Tensor) -> Tensor:
        return self.model(x)

    def get_logits(self, x: Tensor) -> Tensor:
        return self.forward(x)


def deeprobust_wrapper(attack: type[BaseAttack],
                       attack_params: dict,
                       model: nn.Module,
                       inputs: Tensor,
                       labels: Tensor,
                       targets: Optional[Tensor] = None,
                       targeted: bool = False) -> Tensor:
    attack = attack(model=DeepRobustModel(model), device=inputs.device.type)

    attack_param_names = inspect.signature(attack.parse_params).parameters
    if 'classnum' in attack_param_names:  # specify the number of classes for some attacks
        attack_params['classnum'] = [module for module in model.modules()][-1].out_features

    if 'target_label' in inspect.signature(attack.generate).parameters:
        adv_examples = attack.generate(image=inputs, label=labels, target_label=0, **attack_params)
    else:
        adv_examples = attack.generate(image=inputs, label=labels, **attack_params)

    return adv_examples


class DeepRobustMinimalWrapper:
    def __init__(self, model: DeepRobustModel, device: str, attack: type[BaseAttack], init_eps: float,
                 search_steps: int, max_eps: Optional[float] = None):
        self.model = model
        self.attack = attack(model=model, device=device)
        self.init_eps = init_eps
        self.search_steps = search_steps
        self.max_eps = max_eps

    @property
    def parse_params(self):
        return self.attack.parse_params

    def generate(self, image: Tensor, label: Tensor, **attack_params) -> Tensor:
        adv_inputs = image.clone()
        eps_low = torch.full_like(label, 0, dtype=image.dtype)
        best_eps = torch.full_like(eps_low, float('inf') if self.max_eps is None else 2 * self.max_eps)
        found_high = torch.full_like(eps_low, False, dtype=torch.bool)

        eps = torch.full_like(eps_low, self.init_eps)
        for i in range(self.search_steps):
            adv_inputs_run = image.clone()
            for eps_ in torch.unique(eps):
                mask = eps == eps_
                adv_inputs_run[mask] = self.attack.generate(image=image[mask], label=label[mask], epsilon=eps_.item(),
                                                            **attack_params)

            logits = self.model(adv_inputs_run)
            preds = logits.argmax(dim=1)
            is_adv = preds != label

            better_adv = is_adv & (eps < best_eps)
            adv_inputs[better_adv] = adv_inputs_run[better_adv]

            found_high.logical_or_(better_adv)
            eps_low = torch.where(better_adv, eps_low, eps)
            best_eps = torch.where(better_adv, eps, best_eps)

            eps = torch.where(found_high | ((2 * eps_low) >= best_eps), (eps_low + best_eps) / 2, 2 * eps_low)

        return adv_inputs
