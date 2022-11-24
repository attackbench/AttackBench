import inspect
from typing import Optional, Callable
from torch import Tensor, nn


def cleverhans_wrapper(attack: Callable,
                       model: nn.Module,
                       inputs: Tensor,
                       labels: Tensor,
                       targets: Optional[Tensor] = None,
                       targeted: bool = False) -> Tensor:
    attack_kwargs = {}
    attack_parameters = inspect.signature(attack).parameters
    if 'n_classes' in attack_parameters:  # specify the number of classes for some attacks
        attack_kwargs['n_classes'] = [module for module in model.modules()][-1].out_features
    if 'targeted' in attack_parameters:
        attack_kwargs['targeted'] = targeted
    if 'y' in attack_parameters:
        attack_kwargs['y'] = targets if targeted else labels
    if 'y_target' in attack_parameters and targeted:
        attack_kwargs['y_target'] = targets

    adv_examples = attack(model_fn=model, x=inputs, **attack_kwargs)
    return adv_examples
