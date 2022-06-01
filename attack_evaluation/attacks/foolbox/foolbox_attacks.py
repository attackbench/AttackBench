from typing import Optional, Callable
from foolbox.attacks import DatasetAttack, L0FMNAttack, L1FMNAttack, L2FMNAttack, LInfFMNAttack
from foolbox import Misclassification, PyTorchModel, TargetedMisclassification
from foolbox.attacks.base import MinimizationAttack
from foolbox.types import Linf
from torch import Tensor, nn
from cmath import inf
import inspect

_LpFMNAttack = {
    0: L0FMNAttack,
    1: L1FMNAttack,
    2: L2FMNAttack,
    inf: LInfFMNAttack
}


def to_foolbox_model(model: nn.Module) -> PyTorchModel:
    fb_model = PyTorchModel(model=model, bounds=(0, 1))
    return fb_model


def foolbox_wrapper(attack: Callable,
                    model: nn.Module,
                    inputs: Tensor,
                    labels: Tensor,
                    targets: Optional[Tensor] = None,
                    targeted: bool = False) -> Tensor:
    fb_model = to_foolbox_model(model=model)
    if 'inputs' in inspect.signature(attack.func).parameters.keys():
        # this is necessary for dataset attack, which exploits inputs during feed
        fb_attack = attack(model=fb_model, inputs=inputs)
    else:
        fb_attack = attack()
    if targeted:
        criterion = TargetedMisclassification(targets)
    else:
        criterion = Misclassification(labels)

    adv_inputs = fb_attack.run(model=fb_model, inputs=inputs, criterion=criterion)
    return adv_inputs


def fb_lib_dataset_attack(model: PyTorchModel,
                          inputs: Tensor) -> MinimizationAttack:
    attack = DatasetAttack()
    attack.feed(model, inputs)
    return attack


def fb_lib_fmn_attack(norm: float,
                  steps: int,
                  max_stepsize: int,
                  min_stepsize: int,
                  gamma: int,
                  init_attack: Optional[MinimizationAttack],
                  binary_search_steps: int) -> MinimizationAttack:
    attack = _LpFMNAttack[norm](steps=steps, max_stepsize=max_stepsize, min_stepsize=min_stepsize, gamma=gamma,
                                init_attack=init_attack, binary_search_steps=binary_search_steps)
    return attack
