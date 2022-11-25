from functools import partial
from typing import Optional

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
