from typing import Optional
from foolbox.attacks import DatasetAttack
from foolbox import Misclassification, PyTorchModel, TargetedMisclassification
from torch import Tensor, nn


def foolbox_dataset_attack(model: nn.Module,
                   inputs: Tensor,
                   labels: Tensor,
                   targets: Optional[Tensor] = None,
                   targeted: bool = False) -> Tensor:
    fb_model = PyTorchModel(model=model, bounds=(0, 1))
    attack = DatasetAttack()
    attack.feed(fb_model, inputs)
    if targeted:
        criterion = TargetedMisclassification(targets)
    else:
        criterion = Misclassification(labels)

    adv_inputs = attack.run(model=fb_model, inputs=inputs, criterion=criterion)
    return adv_inputs
