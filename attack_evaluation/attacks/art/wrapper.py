import inspect
from typing import Callable, Optional

from art.estimators.classification import PyTorchClassifier
from torch import Tensor, from_numpy, nn


def art_wrapper(attack: Callable,
                model: nn.Module,
                inputs: Tensor,
                labels: Tensor,
                targets: Optional[Tensor] = None,
                targeted: bool = False) -> Tensor:
    loss = nn.CrossEntropyLoss()
    input_shape = inputs.shape[1:]
    output_shape = [module for module in model.modules()][-1].out_features

    art_model = PyTorchClassifier(model=model, clip_values=(0, 1), loss=loss, input_shape=input_shape,
                                  nb_classes=output_shape)
    model_arg = next(iter(inspect.signature(attack.func).parameters))
    attack_kwargs = {model_arg: art_model}

    if 'targeted' in attack.func.attack_params:  # not all attacks have the targeted arg
        attack_kwargs['targeted'] = targeted

    attack = attack(batch_size=len(inputs), **attack_kwargs)
    y = targets if targeted else labels
    adv_inputs = attack.generate(x=inputs.cpu().numpy(), y=y.cpu().numpy())

    return from_numpy(adv_inputs).to(inputs.device)
