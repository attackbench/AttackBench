from functools import partial
from typing import Callable, Optional

from torchattacks import APGD, APGDT, AutoAttack, CW, DeepFool, FAB, FGSM, PGD, PGDL2, SparseFool

from .wrapper import TorchattacksMinimalWrapper
from ..ingredient import minimal_init_eps, minimal_search_steps

_prefix = 'ta'


def ta_apgd():
    name = 'apgd'
    source = 'torchattacks'
    threat_model = 'linf'
    targeted = False  # use a targeted objective for the untargeted attack
    steps = 100
    epsilon = 8 / 255
    num_restarts = 1
    loss = 'ce'
    rho = 0.75


def get_ta_apgd(threat_model: str, targeted: bool, steps: int, epsilon: float, num_restarts: int, loss: str,
                rho: float) -> Callable:
    apgd_func = APGDT if targeted else APGD
    return partial(apgd_func, norm=threat_model.capitalize(), steps=steps, eps=epsilon, n_restarts=num_restarts,
                   loss=loss, rho=rho)


def ta_apgd_minimal():
    name = 'apgd_minimal'
    source = 'torchattacks'
    threat_model = 'linf'
    targeted = False  # use a targeted objective for the untargeted attack
    steps = 100
    num_restarts = 1
    loss = 'ce'
    rho = 0.75


def get_ta_apgd_minimal(threat_model: str, targeted: bool, steps: int, num_restarts: int, loss: str, rho: float,
                        init_eps: Optional[float] = None, search_steps: int = minimal_search_steps) -> Callable:
    init_eps = minimal_init_eps[threat_model] if init_eps is None else init_eps
    max_eps = 1 if threat_model == 'linf' else None
    apgd_func = APGDT if targeted else APGD
    attack = partial(apgd_func, norm=threat_model.capitalize(), steps=steps, n_restarts=num_restarts,
                     loss=loss, rho=rho)
    return partial(TorchattacksMinimalWrapper, attack=attack, init_eps=init_eps, search_steps=search_steps,
                   max_eps=max_eps, batched=True)


def ta_auto_attack():
    name = 'auto_attack'
    source = 'torchattacks'
    threat_model = 'linf'  # available: linf, l2
    epsilon = 0.3
    version = 'standard'


def get_ta_auto_attack(threat_model: str, epsilon: float, version: str) -> Callable:
    return partial(AutoAttack, norm=threat_model.capitalize(), eps=epsilon, version=version)


def ta_cw_l2():
    name = 'cw_l2'
    source = 'torchattacks'
    threat_model = 'l2'
    num_steps = 1000
    c = 0.0001
    kappa = 0
    step_size = 0.01


def get_ta_cw_l2(num_steps: int, c: float, kappa: float, step_size: float) -> Callable:
    return partial(CW, steps=num_steps, c=c, kappa=kappa, lr=step_size)


def ta_deepfool():
    name = 'deepfool'
    source = 'torchattacks'
    threat_model = 'l2'
    num_steps = 50
    overshoot = 0.02


def get_ta_deepfool(num_steps: int, overshoot: float) -> Callable:
    return partial(DeepFool, steps=num_steps, overshoot=overshoot)


def ta_fab():
    name = 'fab'
    source = 'torchattacks'
    threat_model = 'linf'  # available: linf, l2, l1
    num_steps = 100
    epsilon = None
    num_restarts = 1
    alpha_max = 0.1
    eta = 1.05
    beta = 0.9


def get_ta_fab(threat_model: str, num_steps: int, epsilon: Optional[float], num_restarts: int, alpha_max: float,
               eta: float, beta: float) -> Callable:
    return partial(FAB, norm=threat_model.capitalize(), steps=num_steps, eps=epsilon, n_restarts=num_restarts,
                   alpha_max=alpha_max, eta=eta, beta=beta)


def ta_fgsm():
    name = 'fgsm'
    source = 'torchattacks'
    threat_model = 'linf'
    epsilon = 0.007


def get_ta_fgsm(epsilon: float) -> Callable:
    return partial(FGSM, eps=epsilon)


def ta_fgsm_minimal():
    name = 'fgsm_minimal'
    source = 'torchattacks'
    threat_model = 'linf'


def get_ta_fgsm_minimal(init_eps: Optional[float] = None, search_steps: int = minimal_search_steps) -> Callable:
    init_eps = minimal_init_eps['linf'] if init_eps is None else init_eps
    return partial(TorchattacksMinimalWrapper, attack=FGSM, init_eps=init_eps, search_steps=search_steps, max_eps=1,
                   batched=True)


def ta_pgd():
    name = 'pgd'
    source = 'torchattacks'
    threat_model = 'linf'
    num_steps = 40
    epsilon = 0.3
    alpha = 2 / 255
    random_start = True


def get_ta_pgd(num_steps: int, epsilon: float, alpha: float, random_start: bool) -> Callable:
    return partial(PGD, steps=num_steps, eps=epsilon, alpha=alpha, random_start=random_start)


def ta_pgd_minimal():
    name = 'pgd_minimal'
    source = 'torchattacks'
    threat_model = 'linf'
    num_steps = 40
    alpha = 2 / 255
    random_start = True


def get_ta_pgd_minimal(num_steps: int, alpha: float, random_start: bool,
                       init_eps: Optional[float] = None, search_steps: int = minimal_search_steps) -> Callable:
    init_eps = minimal_init_eps['linf'] if init_eps is None else init_eps
    attack = partial(PGD, steps=num_steps, alpha=alpha, random_start=random_start)
    return partial(TorchattacksMinimalWrapper, attack=attack, init_eps=init_eps, search_steps=search_steps, max_eps=1)


def ta_pgd_l2():
    name = 'pgd_l2'
    source = 'torchattacks'
    threat_model = 'l2'
    num_steps = 40
    epsilon = 1.0
    alpha = 0.2
    random_start = True
    eps_for_division = 1e-10


def get_ta_pgd_l2(num_steps: int, epsilon: float, alpha: float, random_start: bool,
                  eps_for_division: float) -> Callable:
    return partial(PGDL2, steps=num_steps, eps=epsilon, alpha=alpha, random_start=random_start,
                   eps_for_division=eps_for_division)


def ta_pgd_l2_minimal():
    name = 'pgd_l2_minimal'
    source = 'torchattacks'
    threat_model = 'l2'
    num_steps = 40
    alpha = 0.2
    random_start = True
    eps_for_division = 1e-10


def get_ta_pgd_l2_minimal(num_steps: int, alpha: float, random_start: bool, eps_for_division: float,
                          init_eps: Optional[float] = None, search_steps: int = minimal_search_steps) -> Callable:
    init_eps = minimal_init_eps['l2'] if init_eps is None else init_eps
    attack = partial(PGDL2, steps=num_steps, alpha=alpha, random_start=random_start, eps_for_division=eps_for_division)
    return partial(TorchattacksMinimalWrapper, attack=attack, init_eps=init_eps, search_steps=search_steps)


def ta_sparsefool():
    name = 'sparsefool'
    source = 'torchattacks'
    threat_model = 'l0'
    num_steps = 20
    lam = 3
    overshoot = 0.02


def get_ta_sparsefool(num_steps: int, lam: float, overshoot: float) -> Callable:
    return partial(SparseFool, steps=num_steps, lam=lam, overshoot=overshoot)
