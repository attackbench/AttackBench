from torchattacks import SparseFool, FAB, FGSM, DeepFool, CW, PGD, PGDL2, AutoAttack
from torchattacks.attack import Attack as TorchAttack
from torch import Tensor, nn
from cmath import inf


def torch_attacks_wrapper(attack: TorchAttack,
                          model: nn.Module,
                          inputs: Tensor,
                          labels: Tensor,
                          **kwargs) -> Tensor:
    attack = attack(model)
    adv_examples = attack(images=inputs, labels=labels)
    return adv_examples


def ta_lib_deepfool(model: nn.Module,
                    norm: float,
                    steps: int,
                    overshoot: float) -> TorchAttack:
    assert norm == 2
    return DeepFool(model=model, steps=steps, overshoot=overshoot)


def ta_lib_sparsefool(model: nn.Module,
                      norm: float,
                      steps: int,
                      lam: float,
                      overshoot: float) -> TorchAttack:
    assert norm == 0
    return SparseFool(model=model, steps=steps, lam=lam, overshoot=overshoot)


def ta_lib_fab(model: nn.Module,
               norm: str,
               eps: float,
               steps: int,
               n_restarts: int,
               alpha_max: float,
               eta: float,
               beta: float,
               targeted: bool) -> TorchAttack:
    assert norm in [1, 2, inf]
    norm = 'L%s' % norm
    n_classes = list(model.modules())[-1].out_features
    return FAB(model=model, steps=steps, norm=norm, eps=eps, n_restarts=n_restarts,
               alpha_max=alpha_max, eta=eta, beta=beta, targeted=targeted, n_classes=n_classes)


def ta_lib_fgsm(model: nn.Module,
                norm: float,
                eps: float) -> TorchAttack:
    assert norm == 0
    return FGSM(model=model, eps=eps)


def ta_lib_cw(model: nn.Module,
              norm: float,
              c: float,
              kappa: float,
              steps: int,
              lr: float) -> TorchAttack:
    assert norm == 2
    return CW(model=model, c=c, kappa=kappa, steps=steps, lr=lr)


def ta_lib_pgd_linf(model: nn.Module,
                    norm: float,
                    eps: float,
                    alpha: float,
                    steps: int,
                    random_start: bool) -> TorchAttack:
    assert norm == inf
    return PGD(model=model, eps=eps, alpha=alpha, steps=steps, random_start=random_start)


def ta_lib_pgd_l2(model: nn.Module,
                  norm: float,
                  eps: float,
                  alpha: float,
                  steps: int,
                  random_start: bool,
                  eps_for_division: float) -> TorchAttack:
    assert norm == 2
    return PGDL2(model=model, eps=eps, alpha=alpha, steps=steps,
                 random_start=random_start, eps_for_division=eps_for_division)


def ta_lib_auto_attack(model: nn.Module,
                       norm: float,
                       eps: float,
                       version: str) -> TorchAttack:
    assert norm == [2, inf]
    n_classes = list(model.modules())[-1].out_features
    return AutoAttack(model=model, eps=eps, version=version, n_classes=n_classes)
