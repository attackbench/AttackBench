from torchattacks import SparseFool
from torchattacks.attack import Attack as TorchAttack
from torch import Tensor, nn


def torch_attacks_wrapper(attack: TorchAttack,
                          model: nn.Module,
                          inputs: Tensor,
                          labels: Tensor,
                          **kwargs) -> Tensor:
    attack = attack(model)
    adv_examples = attack(images=inputs, labels=labels)
    return adv_examples


def sparsefool(model: nn.Module,
               steps: int,
               lam: float,
               overshoot: float) -> TorchAttack:
    return SparseFool(model=model, steps=steps, lam=lam, overshoot=overshoot)
