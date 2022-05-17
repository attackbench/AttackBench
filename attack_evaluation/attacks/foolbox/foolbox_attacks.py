from typing import Optional
from foolbox import Misclassification, PyTorchModel, TargetedMisclassification
import foolbox.attacks
from torch import Tensor, nn
from inspect import signature

_pgd_attacks = {
    2: "L2ProjectedGradientDescentAttack",
    'inf': "LinfProjectedGradientDescentAttack",
}

_deepfool_attacks = {
    2: "L2DeepFoolAttack",
    'inf': "LinfDeepFoolAttack",
}

_bb_attacks = {
    0: "L0BrendelBethgeAttack",
    1: "L1BrendelBethgeAttack",
    2: "L2BrendelBethgeAttack",
    'inf': "LinfinityBrendelBethgeAttack",
}

_fmn_attacks = {
    0: "L0FMNAttack",
    1: "L1FMNAttack",
    2: "L2FMNAttack",
    'inf': "LInfFMNAttack",
}

_foolbox_attacks = {
    "pgd": _pgd_attacks,
    "deepfool": _deepfool_attacks,
    "fmn": _fmn_attacks
}


def foolbox_attack(model: nn.Module,
                   inputs: Tensor,
                   labels: Tensor,
                   targets: Optional[Tensor] = None,
                   targeted: bool = False,
                   name: str = "pgd",
                   norm: float = 2,
                   eps: float = 0.3, **kwargs) -> Tensor:
    attack_class = str(_foolbox_attacks[name][norm])
    attack = getattr(foolbox.attacks, attack_class)(**kwargs)
    fb_model = PyTorchModel(model=model, bounds=(0, 1))
    if targeted:
        criterion = TargetedMisclassification(targets)
    else:
        criterion = Misclassification(labels)
    if 'epsilon' in signature(attack.run).parameters:
        adv_inputs = attack.run(model=fb_model, inputs=inputs, criterion=criterion, epsilon=eps)
    else:
        # minimum distance attacks
        adv_inputs = attack.run(model=fb_model, inputs=inputs, criterion=criterion)
    return adv_inputs


