import inspect
from typing import Optional, Callable
from torch import Tensor, nn


def cleverhans_wrapper(attack: Callable,
                       model: nn.Module,
                       inputs: Tensor,
                       labels: Tensor,
                       targets: Optional[Tensor] = None,
                       targeted: bool = False) -> Tensor:

    if 'n_classes' in inspect.signature(attack).parameters:  # specify the number of classes for some attacks
        n_classes = [module for module in model.modules()][-1].out_features
        adv_examples = attack(model_fn=model, x=inputs, n_classes=n_classes)
    else:
        adv_examples = attack(model_fn=model, x=inputs)

    return adv_examples
