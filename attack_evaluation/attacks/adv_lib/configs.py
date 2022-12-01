from functools import partial
from typing import Callable, Optional

from adv_lib.attacks import (
    alma,
    apgd,
    apgd_targeted,
    carlini_wagner_l2,
    carlini_wagner_linf,
    ddn,
    fab,
    fmn,
    pdgd,
    pdpgd,
    pgd_linf,
    tr,
    vfga,
)

from .wrapper import adv_lib_minimal_wrapper, adv_lib_wrapper

_prefix = 'adv_lib'
_wrapper = adv_lib_wrapper
_norms = {
    'l0': 0,
    'l1': 1,
    'l2': 2,
    'linf': float('inf'),
}


def adv_lib_alma_l1():
    name = 'alma'
    source = 'adv_lib'
    threat_model = 'l1'
    num_steps = 1000
    alpha = 0.9
    init_lr_distance = 0.5


def adv_lib_alma_l2():
    name = 'alma'
    source = 'adv_lib'
    threat_model = 'l2'
    num_steps = 1000
    alpha = 0.9
    init_lr_distance = 0.1


def get_adv_lib_alma(threat_model: str, num_steps: int, alpha: float, init_lr_distance: float) -> Callable:
    return partial(alma, distance=threat_model, num_steps=num_steps, α=alpha, init_lr_distance=init_lr_distance)


def adv_lib_apgd_l1():
    name = 'apgd'
    source = 'adv_lib'
    threat_model = 'l1'
    epsilon = 10
    targeted = False  # use a targeted objective for the untargeted attack
    num_steps = 100
    num_restarts = 1
    loss_function = 'dlr'
    rho = 0.75
    use_large_reps = True
    use_rs = True


def adv_lib_apgd_l2():
    name = 'apgd'
    source = 'adv_lib'
    threat_model = 'l2'
    epsilon = 1
    targeted = False  # use a targeted objective for the untargeted attack
    num_steps = 100
    num_restarts = 1
    loss_function = 'dlr'
    rho = 0.75
    use_large_reps = False
    use_rs = True


def adv_lib_apgd_linf():
    name = 'apgd'
    source = 'adv_lib'
    threat_model = 'linf'
    epsilon = 4 / 255
    targeted = False  # use a targeted objective for the untargeted attack
    num_steps = 100
    num_restarts = 1
    loss_function = 'dlr'
    rho = 0.75
    use_large_reps = False
    use_rs = True


def get_adv_lib_apgd(threat_model: str, epsilon: float, targeted: bool, num_steps: int, num_restarts: int,
                     loss_function: str, rho: float, use_large_reps: bool, use_rs: bool) -> Callable:
    attack_func = apgd_targeted if targeted else apgd
    return partial(attack_func, norm=_norms[threat_model], eps=epsilon, n_iter=num_steps, n_restarts=num_restarts,
                   loss_function=loss_function, rho=rho, use_large_reps=use_large_reps, use_rs=use_rs)


def adv_lib_apgd_minimal_l1():
    name = 'apgd_minimal'
    source = 'adv_lib'
    threat_model = 'l1'
    targeted = False  # use a targeted objective for the untargeted attack
    num_steps = 100
    num_restarts = 1
    loss_function = 'dlr'
    rho = 0.75
    use_large_reps = True
    use_rs = True

    init_eps = 10  # initial guess for line search
    search_steps = 20  # number of search steps for line + binary search


def adv_lib_apgd_minimal_l2():
    name = 'apgd_minimal'
    source = 'adv_lib'
    threat_model = 'l2'
    targeted = False  # use a targeted objective for the untargeted attack
    num_steps = 100
    num_restarts = 1
    loss_function = 'dlr'
    rho = 0.75
    use_large_reps = False
    use_rs = True

    init_eps = 1  # initial guess for line search
    search_steps = 20  # number of search steps for line + binary search


def adv_lib_apgd_minimal_linf():
    name = 'apgd_minimal'
    source = 'adv_lib'
    threat_model = 'linf'
    targeted = False  # use a targeted objective for the untargeted attack
    num_steps = 100
    num_restarts = 1
    loss_function = 'dlr'
    rho = 0.75
    use_large_reps = False
    use_rs = True

    init_eps = 1 / 255  # initial guess for line search
    search_steps = 20  # number of search steps for line + binary search


def get_adv_lib_apgd_minimal(threat_model: str, targeted: bool, num_steps: int, num_restarts: int, loss_function: str,
                             rho: float, use_large_reps: bool, use_rs: bool,
                             init_eps: float, search_steps: int) -> Callable:
    attack_func = apgd_targeted if targeted else apgd
    attack = partial(attack_func, norm=_norms[threat_model], n_iter=num_steps, n_restarts=num_restarts,
                     loss_function=loss_function, rho=rho, use_large_reps=use_large_reps, use_rs=use_rs)

    max_eps = 1 if threat_model == 'linf' else None
    return partial(adv_lib_minimal_wrapper, attack=attack, init_eps=init_eps, max_eps=max_eps,
                   search_steps=search_steps)


def adv_lib_cw_l2():
    name = 'cw_l2'
    source = 'adv_lib'
    threat_model = 'l2'
    confidence = 0
    step_size = 0.01
    initial_const = 0.001
    num_binary_search_steps = 9
    num_steps = 100  # default was 10000
    abort_early = True


def get_adv_lib_cw_l2(confidence: float, step_size: float, initial_const: float, num_binary_search_steps: int,
                      num_steps: int, abort_early: bool) -> Callable:
    return partial(carlini_wagner_l2, confidence=confidence, learning_rate=step_size, initial_const=initial_const,
                   binary_search_steps=num_binary_search_steps, max_iterations=num_steps, abort_early=abort_early)


def adv_lib_cw_linf():
    name = 'cw_linf'
    source = 'adv_lib'
    threat_model = 'linf'
    step_size = 0.01
    num_steps = 1000
    initial_const = 1e-5
    largest_const = 2e+1
    const_factor = 2
    reduce_const = False
    decrease_factor = 0.9
    abort_early = True


def get_adv_lib_cw_linf(step_size: float, num_steps: int, initial_const: float, largest_const: float,
                        const_factor: float, reduce_const: bool, decrease_factor: float, abort_early: bool) -> Callable:
    return partial(carlini_wagner_linf, learning_rate=step_size, max_iterations=num_steps,
                   initial_const=initial_const, largest_const=largest_const, const_factor=const_factor,
                   reduce_const=reduce_const, decrease_factor=decrease_factor, abort_early=abort_early)


def adv_lib_ddn():
    name = 'ddn'
    source = 'adv_lib'
    threat_model = 'l2'
    num_steps = 1000
    init_norm = 1
    gamma = 0.05


def get_adv_lib_ddn(num_steps: int, gamma: float, init_norm: float) -> Callable:
    return partial(ddn, steps=num_steps, γ=gamma, init_norm=init_norm)


def adv_lib_fab():
    name = 'fab'
    source = 'adv_lib'
    threat_model = 'linf'
    num_steps = 100
    epsilon = None
    alpha_max = 0.1
    beta = 0.9
    eta = 1.05
    num_restarts = None
    targeted_restarts = False


def get_adv_lib_fab(threat_model: str, num_steps: int, epsilon: Optional[float], alpha_max: float, beta: float,
                    eta: float, num_restarts: Optional[int], targeted_restarts: bool) -> Callable:
    return partial(fab, norm=_norms[threat_model], n_iter=num_steps, ε=epsilon, α_max=alpha_max, β=beta, η=eta,
                   restarts=num_restarts, targeted_restarts=targeted_restarts)


def adv_lib_fmn():
    name = 'fmn'
    source = 'adv_lib'
    threat_model = 'linf'
    num_steps = 1000
    max_stepsize = 1
    gamma = 0.05


def get_adv_lib_fmn(threat_model: str, num_steps: int, max_stepsize: float, gamma: float) -> Callable:
    return partial(fmn, norm=_norms[threat_model], steps=num_steps, α_init=max_stepsize, γ_init=gamma)


def adv_lib_pdgd():
    name = 'pdgd'
    source = 'adv_lib'
    threat_model = 'l2'
    num_steps = 500
    random_init = 0
    primal_lr = 0.1
    primal_lr_decrease = 0.01
    dual_ratio_init = 0.01
    dual_lr = 0.1
    dual_lr_decrease = 0.1
    dual_ema = 0.9
    dual_min_ratio = 1e-6


def get_adv_lib_pdgd(num_steps: int, random_init: float, primal_lr: float, primal_lr_decrease: float,
                     dual_ratio_init: float, dual_lr: float, dual_lr_decrease: float, dual_ema: float,
                     dual_min_ratio: float) -> Callable:
    return partial(pdgd, num_steps=num_steps, random_init=random_init, primal_lr=primal_lr,
                   primal_lr_decrease=primal_lr_decrease, dual_ratio_init=dual_ratio_init, dual_lr=dual_lr,
                   dual_lr_decrease=dual_lr_decrease, dual_ema=dual_ema, dual_min_ratio=dual_min_ratio)


def adv_lib_pdpgd():
    name = 'pdpgd'
    source = 'adv_lib'
    threat_model = 'linf'
    num_steps = 500
    random_init = 0
    proximal_operator = None
    primal_lr = 0.1
    primal_lr_decrease = 0.01
    dual_ratio_init = 0.01
    dual_lr = 0.1
    dual_lr_decrease = 0.1
    dual_ema = 0.9
    dual_min_ratio = 1e-6
    proximal_steps = 5
    ε_threshold = 1e-2


def get_adv_lib_pdpgd(threat_model: str, num_steps: int, random_init: float, proximal_operator: Optional[float],
                      primal_lr: float, primal_lr_decrease: float, dual_ratio_init: float, dual_lr: float,
                      dual_lr_decrease: float, dual_ema: float, dual_min_ratio: float, proximal_steps: int,
                      ε_threshold: float) -> Callable:
    return partial(pdpgd, norm=_norms[threat_model], num_steps=num_steps, random_init=random_init,
                   proximal_operator=proximal_operator, primal_lr=primal_lr, primal_lr_decrease=primal_lr_decrease,
                   dual_ratio_init=dual_ratio_init, dual_lr=dual_lr, dual_lr_decrease=dual_lr_decrease,
                   dual_ema=dual_ema, dual_min_ratio=dual_min_ratio, proximal_steps=proximal_steps,
                   ε_threshold=ε_threshold)


def adv_lib_pgd():
    name = 'pgd'
    source = 'adv_lib'
    threat_model = 'linf'
    epsilon = 4 / 255
    num_steps = 40
    random_init = True
    num_restarts = 1
    loss_function = 'ce'
    relative_step_size = 0.01 / 0.3
    absolute_step_size = None


def get_adv_lib_pgd(epsilon: float, num_steps: int, random_init: bool, num_restarts: int, loss_function: str,
                    relative_step_size: float, absolute_step_size: Optional[float]):
    return partial(pgd_linf, ε=epsilon, steps=num_steps, random_init=random_init, restarts=num_restarts,
                   loss_function=loss_function, relative_step_size=relative_step_size,
                   absolute_step_size=absolute_step_size)


def adv_lib_pgd_minimal():
    name = 'pgd_minimal'
    source = 'adv_lib'
    threat_model = 'linf'
    num_steps = 40
    random_init = True
    num_restarts = 1
    loss_function = 'ce'
    relative_step_size = 0.01 / 0.3
    absolute_step_size = None

    init_eps = 1 / 255  # initial guess for line search
    search_steps = 20  # number of search steps for line + binary search


def get_adv_lib_pgd_minimal(num_steps: int, random_init: bool, num_restarts: int, loss_function: str,
                            relative_step_size: float, absolute_step_size: Optional[float],
                            init_eps: float, search_steps: int) -> Callable:
    attack = partial(pgd_linf, steps=num_steps, random_init=random_init, restarts=num_restarts,
                     loss_function=loss_function, relative_step_size=relative_step_size,
                     absolute_step_size=absolute_step_size)

    return partial(adv_lib_minimal_wrapper, attack=attack, init_eps=init_eps, max_eps=1, search_steps=search_steps,
                   eps_name='ε')


def adv_lib_tr():
    name = 'tr'
    source = 'adv_lib'
    threat_model = 'linf'
    num_steps = 100
    adaptive = False
    epsilon = 0.001
    c = 9
    worst_case = False


def get_adv_lib_tr(threat_model: str, num_steps: int, adaptive: bool, epsilon: float, c: int,
                   worst_case: bool) -> Callable:
    return partial(tr, p=_norms[threat_model], iter=num_steps, adaptive=adaptive, eps=epsilon, c=c,
                   worst_case=worst_case)


def adv_lib_vfga():
    name = 'vfga'
    source = 'adv_lib'
    threat_model = 'l0'
    num_steps = None
    n_samples = 10
    large_memory = False


def get_adv_lib_vfga(num_steps: int, n_samples: int, large_memory: bool) -> Callable:
    return partial(vfga, max_iter=num_steps, n_samples=n_samples, large_memory=large_memory)
