from functools import partial
from typing import Callable, Optional

from torchattacks import APGD, APGDT, AutoAttack, CW, DeepFool, FAB, FGSM, PGD, PGDL2, SparseFool

from ..utils import ConfigGetter

_norms = {
    1: 'L1',
    2: 'L2',
    float('inf'): 'Linf'
}


def ta_apgd():
    name = 'apgd'
    source = 'torchattacks'  # available: ['torchattacks']
    norm = float('inf')
    targeted = False  # use a targeted objective for the untargeted attack
    steps = 100
    epsilon = 8 / 255
    num_restarts = 1
    loss = 'ce'
    rho = 0.75


def get_ta_apgd(norm: float, targeted: bool, steps: int, epsilon: float, num_restarts: int, loss: str,
                rho: float) -> Callable:
    apgd_func = APGDT if targeted else APGD
    return partial(apgd_func, norm=_norms[float(norm)], steps=steps, eps=epsilon, n_restarts=num_restarts, loss=loss,
                   rho=rho)


def ta_auto_attack():
    name = 'auto_attack'
    source = 'torchattacks'  # available: ['torchattacks']
    norm = float('inf')  # available: inf, 2
    epsilon = 0.3
    version = 'standard'


def get_ta_auto_attack(norm: float, epsilon: float, version: str) -> Callable:
    return partial(AutoAttack, norm=_norms[float(norm)], eps=epsilon, version=version)


def ta_cw_l2():
    name = 'cw_l2'
    source = 'torchattacks'  # available: ['torchattacks']
    num_steps = 1000
    c = 0.0001
    kappa = 0
    step_size = 0.01


def get_ta_cw_l2(num_steps: int, c: float, kappa: float, step_size: float) -> Callable:
    return partial(CW, steps=num_steps, c=c, kappa=kappa, lr=step_size)


def ta_deepfool():
    name = 'deepfool'
    source = 'torchattacks'  # available: ['torchattacks', 'art']
    num_steps = 50
    overshoot = 0.02


def get_ta_deepfool(num_steps: int, overshoot: float) -> Callable:
    return partial(DeepFool, steps=num_steps, overshoot=overshoot)


def ta_fab():
    name = 'fab'
    source = 'torchattacks'  # available: ['torchattacks']
    norm = float('inf')  # available: inf, 2, 1
    num_steps = 100
    epsilon = None
    num_restarts = 1
    alpha_max = 0.1
    eta = 1.05
    beta = 0.9
    targeted = False


def get_ta_fab(norm: float, num_steps: int, epsilon: Optional[float], num_restarts: int, alpha_max: float, eta: float,
               beta: float, targeted: bool) -> Callable:
    return partial(FAB, norm=_norms[float(norm)], steps=num_steps, eps=epsilon, n_restarts=num_restarts,
                   alpha_max=alpha_max,
                   eta=eta, beta=beta, targeted=targeted)


def ta_fgsm():
    name = 'fgsm'
    source = 'torchattacks'  # available: ['torchattacks']
    epsilon = 0.007


def get_ta_fgsm(epsilon: float) -> Callable:
    return partial(FGSM, eps=epsilon)


def ta_pgd():
    name = 'pgd'
    source = 'torchattacks'  # available: ['torchattacks']
    num_steps = 40
    epsilon = 0.3
    alpha = 2 / 255
    random_start = True


def get_ta_pgd(num_steps: int, epsilon: float, alpha: float, random_start: bool) -> Callable:
    return partial(PGD, steps=num_steps, eps=epsilon, alpha=alpha, random_start=random_start)


def ta_pgd_l2():
    name = 'pgd_l2'
    source = 'torchattacks'  # available: ['torchattacks']
    num_steps = 40
    epsilon = 1.0
    alpha = 0.2
    random_start = True
    eps_for_division = 1e-10


def get_ta_pgd_l2(num_steps: int, epsilon: float, alpha: float, random_start: bool,
                  eps_for_division: float) -> Callable:
    return partial(PGDL2, steps=num_steps, eps=epsilon, alpha=alpha, random_start=random_start,
                   eps_for_division=eps_for_division)


def ta_sparsefool():
    name = 'sparsefool'
    source = 'torchattacks'  # available: ['torchattacks']
    num_steps = 20
    lam = 3
    overshoot = 0.02


def get_ta_sparsefool(num_steps: int, lam: float, overshoot: float) -> Callable:
    return partial(SparseFool, steps=num_steps, lam=lam, overshoot=overshoot)


torchattacks_index = {
    'apgd': ConfigGetter(config=ta_apgd, getter=get_ta_apgd),
    'auto_attack': ConfigGetter(config=ta_auto_attack, getter=get_ta_auto_attack),
    'cw_l2': ConfigGetter(config=ta_cw_l2, getter=get_ta_cw_l2),
    'deepfool': ConfigGetter(config=ta_deepfool, getter=get_ta_deepfool),
    'fab': ConfigGetter(config=ta_fab, getter=get_ta_fab),
    'fgsm': ConfigGetter(config=ta_fgsm, getter=get_ta_fgsm),
    'pgd': ConfigGetter(config=ta_pgd, getter=get_ta_pgd),
    'pgd_l2': ConfigGetter(config=ta_pgd_l2, getter=get_ta_pgd_l2),
    'sparsefool': ConfigGetter(config=ta_sparsefool, getter=get_ta_sparsefool),
}
