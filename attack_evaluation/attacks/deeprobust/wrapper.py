import inspect
from typing import Optional

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


def deeprobust_wrapper(attack: BaseAttack,
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
