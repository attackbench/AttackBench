from typing import Callable, Optional

from torch import Tensor, nn


def adv_lib_wrapper(attack: Callable,
                    model: nn.Module,
                    inputs: Tensor,
                    labels: Tensor,
                    targets: Optional[Tensor] = None,
                    targeted: bool = False) -> Tensor:
    attack_labels = targets if targeted else labels
    return attack(model=model, inputs=inputs, labels=attack_labels, targeted=targeted)
