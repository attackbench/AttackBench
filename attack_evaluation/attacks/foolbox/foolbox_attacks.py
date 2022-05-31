from typing import Optional
from foolbox.attacks import DatasetAttack
from foolbox import Misclassification, PyTorchModel, TargetedMisclassification
from torch import Tensor, nn


def to_foolbox_model(model: nn.Module) -> PyTorchModel:
    fb_model = PyTorchModel(model=model, bounds=(0, 1))
    return fb_model


def foolbox_run(attack,
                fb_model: PyTorchModel,
                inputs: Tensor,
                labels: Tensor,
                targets: Optional[Tensor] = None,
                targeted: bool = False) -> Tensor:
    if targeted:
        criterion = TargetedMisclassification(targets)
    else:
        criterion = Misclassification(labels)

    adv_inputs = attack.run(model=fb_model, inputs=inputs, criterion=criterion)
    return adv_inputs


def foolbox_dataset_attack(model: nn.Module,
                           inputs: Tensor,
                           labels: Tensor,
                           targets: Optional[Tensor] = None,
                           targeted: bool = False) -> Tensor:
    fb_model = to_foolbox_model(model=model)
    attack = DatasetAttack()
    attack.feed(fb_model, inputs)
    return foolbox_run(attack, fb_model, inputs, labels, targets, targeted)
