from typing import Callable, Optional

from foolbox import Misclassification, PyTorchModel, TargetedMisclassification
from torch import Tensor, nn


def foolbox_wrapper(attack: Callable,
                    model: nn.Module,
                    inputs: Tensor,
                    labels: Tensor,
                    targets: Optional[Tensor] = None,
                    targeted: bool = False) -> Tensor:
    fb_model = PyTorchModel(model=model, bounds=(0, 1))
    fb_attack = attack()

    if hasattr(fb_attack, 'feed'):
        fb_attack.feed(model=fb_model, inputs=inputs)

    if targeted:
        criterion = TargetedMisclassification(targets)
    else:
        criterion = Misclassification(labels)

    adv_inputs = fb_attack.run(model=fb_model, inputs=inputs, criterion=criterion)
    return adv_inputs
