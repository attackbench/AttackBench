import math
from abc import abstractmethod
from typing import Any, Optional, Tuple, Union

import eagerpy as ep
from eagerpy.astensor import T
from foolbox import Misclassification, Model, PyTorchModel, TargetedMisclassification
import foolbox.attacks
from torch import Tensor, nn

_pgd_attacks = {
    2: "L2ProjectedGradientDescentAttack",
    float('inf'): "LinfProjectedGradientDescentAttack",
}

_deepfool_attacks = {
    2: "L2DeepFoolAttack",
    float('inf'): "LinfDeepFoolAttack",
}

_bb_attacks = {
    0: "L0BrendelBethgeAttack",
    1: "L1BrendelBethgeAttack",
    2: "L2BrendelBethgeAttack",
    float('inf'): "LinfinityBrendelBethgeAttack",
}

_fmn_attacks = {
    0: "L0FMNAttack",
    1: "L1FMNAttack",
    2: "L2FMNAttack",
    float('inf'): "LInfFMNAttack",
}

_foolbox_attacks = {
    "pgd": _pgd_attacks,
    "deepfool": _deepfool_attacks,
    "BB": _bb_attacks,
    "fmn": _fmn_attacks
}


def foolbox_attack(model: nn.Module,
                   inputs: Tensor,
                   labels: Tensor,
                   targets: Optional[Tensor] = None,
                   targeted: bool = False,
                   attack: str = "pgd",
                   norm: float = 2, **kwargs) -> Tensor:
    name = str(_foolbox_attacks[attack][norm])
    attack = getattr(foolbox.attacks, name, None)(**kwargs)
    fb_model = PyTorchModel(model=model, bounds=(0, 1))
    if targeted:
        criterion = TargetedMisclassification(targets)
    else:
        criterion = Misclassification(labels)
    adv_inputs = attack.run(model=fb_model, inputs=inputs, criterion=criterion)
    return adv_inputs
