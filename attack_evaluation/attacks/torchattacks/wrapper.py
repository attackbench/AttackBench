from typing import Optional

from torch import Tensor, nn
from torchattacks.attack import Attack


def torchattacks_wrapper(attack: Attack,
                         model: nn.Module,
                         inputs: Tensor,
                         labels: Tensor,
                         targets: Optional[Tensor] = None,
                         targeted: bool = False) -> Tensor:
    attack = attack(model=model)
    if targeted:
        attack.set_mode_targeted_by_function()
    adv_examples = attack(inputs=inputs, labels=targets if targeted else labels)
    return adv_examples
