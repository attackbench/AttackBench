import math
from abc import abstractmethod
from typing import Any, Optional, Tuple, Union

import eagerpy as ep
from eagerpy.astensor import T
from foolbox import Misclassification, Model, PyTorchModel, TargetedMisclassification
from foolbox.attacks import L2ProjectedGradientDescentAttack, LinfProjectedGradientDescentAttack
from torch import Tensor, nn

_pgd_attacks = {
    2: L2ProjectedGradientDescentAttack,
    float('inf'): LinfProjectedGradientDescentAttack,
}


def fmn_attack(model: nn.Module,
               inputs: Tensor,
               labels: Tensor,
               targets: Optional[Tensor] = None,
               targeted: bool = False,
               norm: float = 2, **kwargs) -> Tensor:
    attack = _pgd_attacks[norm](**kwargs)
    fb_model = PyTorchModel(model=model, bounds=(0, 1))
    if targeted:
        criterion = TargetedMisclassification(targets)
    else:
        criterion = Misclassification(labels)
    adv_inputs = attack.run(model=fb_model, inputs=inputs, criterion=criterion)
    return adv_inputs
