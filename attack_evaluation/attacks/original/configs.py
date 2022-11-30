from functools import partial
from typing import Callable, Optional

from .auto_pgd import apgd_attack, apgd_t_attack
from .deepfool import deepfool_attack
from .fast_adaptive_boundary import fab_attack
from .fast_minimum_norm import fmn_attack
from .trust_region import tr_attack

_prefix = 'original'


def _wrapper(attack, **kwargs): return attack(**kwargs)


def original_apgd():
    name = 'apgd'
    source = 'original'
    threat_model = 'linf'
    num_steps = 100
    num_restarts = 1
    epsilon = 0.3
    loss = 'ce'  # loss function in ['ce', 'dlr']
    rho = .75
    use_largereps = False  # set True with L1 norm


def get_original_apgd(threat_model: str, num_steps: int, num_restarts: int, epsilon: float, loss: str, rho: float,
                      use_largereps: bool) -> Callable:
    return partial(apgd_attack, threat_model=threat_model, n_iter=num_steps, n_restarts=num_restarts, eps=epsilon,
                   loss=loss, rho=rho, use_largereps=use_largereps)


def original_apgd_t():
    name = 'apgd_t'
    source = 'original'
    threat_model = 'linf'
    num_steps = 100
    num_restarts = 1
    num_target_classes = 9
    epsilon = 0.3
    rho = .75
    use_largereps = False  # set True with L1 norm


def get_original_apgd_t(threat_model: str, num_steps: int, num_restarts: int, num_target_classes: int, epsilon: float,
                        rho: float, use_largereps: bool) -> Callable:
    return partial(apgd_t_attack, threat_model=threat_model, n_iter=num_steps, n_restarts=num_restarts,
                   n_target_classes=num_target_classes, eps=epsilon, rho=rho, use_largereps=use_largereps)


def original_deepfool():
    name = 'deepfool'
    source = 'original'
    threat_model = 'l2'
    num_classes = 10  # number of classes to test gradient (can be different from the number of classes of the model)
    overshoot = 0.02
    num_steps = 50


def get_original_deepfool(num_classes: int, overshoot: float, num_steps: int) -> Callable:
    return partial(deepfool_attack, num_classes=num_classes, overshoot=overshoot, max_iter=num_steps)


def original_fab():
    name = 'fab'
    source = 'original'
    threat_model = 'linf'
    num_restarts = 1
    num_steps = 100
    epsilon = None
    alpha_max = 0.1
    eta = 1.05
    beta = 0.9
    targeted_variant = False
    n_target_classes = 9


def get_original_fab(threat_model: str, num_restarts: int, num_steps: int, epsilon: Optional[float], alpha_max: float,
                     eta: float, beta: float, targeted_variant: bool, n_target_classes: int) -> Callable:
    return partial(fab_attack, threat_model=threat_model, n_restarts=num_restarts, n_iter=num_steps, eps=epsilon,
                   alpha_max=alpha_max, eta=eta, beta=beta, targeted_variant=targeted_variant,
                   n_target_classes=n_target_classes)


def original_fmn():
    name = 'fmn'
    source = 'original'  # available: ['original', 'adv_lib']
    threat_model = 'linf'
    num_steps = 1000
    max_step_size = 1
    gamma = 0.05


def get_original_fmn(threat_model: str, num_steps: int, max_step_size: float, gamma: float) -> Callable:
    return partial(fmn_attack, threat_model=threat_model, steps=num_steps, max_stepsize=max_step_size, gamma=gamma)


def original_tr():
    name = 'tr'
    source = 'original'
    threat_model = 'linf'
    adaptive = False
    epsilon = 0.001
    c = 9
    num_steps = 100


def get_original_tr(threat_model: str, adaptive: bool, epsilon: float, c: int, num_steps: int) -> Callable:
    return partial(tr_attack, threat_model=threat_model, adaptive=adaptive, eps=epsilon, c=c, iter=num_steps)
