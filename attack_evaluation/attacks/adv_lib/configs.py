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

from ..utils import ConfigGetter


def adv_lib_alma():
    name = 'alma'
    source = 'adv_lib'  # available: ['adv_lib']
    distance = 'l2'
    steps = 1000
    alpha = 0.9
    init_lr_distance = 1


def get_adv_lib_alma(distance: float, steps: int, alpha: float, init_lr_distance: float) -> Callable:
    return partial(alma, distance=distance, num_steps=steps, α=alpha, init_lr_distance=init_lr_distance)


def adv_lib_apgd():
    name = 'apgd'
    source = 'adv_lib'  # available: ['adv_lib']
    norm = float('inf')
    eps = 4 / 255
    targeted = False  # use a targeted objective for the untargeted attack
    n_iter = 100
    n_restarts = 1
    loss_function = 'dlr'
    rho = 0.75
    use_large_reps = False
    use_rs = True


def get_adv_lib_apgd(norm: float, eps: float, targeted: bool, n_iter: int, n_restarts: int, loss_function: str,
                     rho: float, use_large_reps: bool, use_rs: bool) -> Callable:
    attack_func = apgd_targeted if targeted else apgd
    return partial(attack_func, norm=norm, eps=eps, n_iter=n_iter, n_restarts=n_restarts, loss_function=loss_function,
                   rho=rho, use_large_reps=use_large_reps, use_rs=use_rs)


def adv_lib_cw_l2():
    name = 'cw_l2'
    source = 'adv_lib'  # available: ['adv_lib']
    confidence = 0
    learning_rate = 0.01
    initial_const = 0.001
    binary_search_steps = 9
    max_iterations = 10000
    abort_early = True


def get_adv_lib_cw_l2(confidence: float, learning_rate: float, initial_const: float, binary_search_steps: int,
                      max_iterations: int, abort_early: bool) -> Callable:
    return partial(carlini_wagner_l2, confidence=confidence, learning_rate=learning_rate, initial_const=initial_const,
                   binary_search_steps=binary_search_steps, max_iterations=max_iterations, abort_early=abort_early)


def adv_lib_cw_linf():
    name = 'cw_linf'
    source = 'adv_lib'  # available: ['adv_lib']
    learning_rate = 0.01
    max_iterations = 1000
    initial_const = 1e-5
    largest_const = 2e+1
    const_factor = 2
    reduce_const = False
    decrease_factor = 0.9
    abort_early = True


def get_adv_lib_cw_linf(learning_rate: float, max_iterations: int, initial_const: float, largest_const: float,
                        const_factor: float, reduce_const: bool, decrease_factor: float, abort_early: bool) -> Callable:
    return partial(carlini_wagner_linf, learning_rate=learning_rate, max_iterations=max_iterations,
                   initial_const=initial_const, largest_const=largest_const, const_factor=const_factor,
                   reduce_const=reduce_const, decrease_factor=decrease_factor, abort_early=abort_early)


def adv_lib_ddn():
    name = 'ddn'
    source = 'adv_lib'  # available: ['adv_lib']
    steps = 1000
    init_norm = 1
    gamma = 0.05


def get_adv_lib_ddn(steps: int, gamma: float, init_norm: float) -> Callable:
    return partial(ddn, steps=steps, γ=gamma, init_norm=init_norm)


def adv_lib_fab():
    name = 'fab'
    source = 'adv_lib'  # available: ['adv_lib']
    norm = float('inf')
    n_iter = 100
    alpha_max = 0.1
    beta = 0.9
    eta = 1.05
    restarts = None
    targeted_restarts = False


def get_adv_lib_fab(norm: float, n_iter: int, alpha_max: float, beta: float, eta: float, restarts: Optional[int],
                    targeted_restarts: bool) -> Callable:
    return partial(fab, norm=norm, n_iter=n_iter, α_max=alpha_max, β=beta, η=eta, restarts=restarts,
                   targeted_restarts=targeted_restarts)


def adv_lib_fmn():
    name = 'fmn'
    source = 'adv_lib'  # available: ['original', 'adv_lib']
    norm = 2
    steps = 1000
    max_stepsize = 1
    gamma = 0.05


def get_adv_lib_fmn(norm: float, steps: int, max_stepsize: float, gamma: float) -> Callable:
    return partial(fmn, norm=norm, steps=steps, α_init=max_stepsize, γ_init=gamma)


def adv_lib_pdgd():
    name = 'pdgd'
    source = 'adv_lib'  # available: ['adv_lib']
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
    source = 'adv_lib'  # available: ['adv_lib']
    norm = float('inf')
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


def get_adv_lib_pdpgd(norm: float, num_steps: int, random_init: float, proximal_operator: Optional[float],
                      primal_lr: float, primal_lr_decrease: float, dual_ratio_init: float, dual_lr: float,
                      dual_lr_decrease: float, dual_ema: float, dual_min_ratio: float, proximal_steps: int,
                      ε_threshold: float) -> Callable:
    return partial(pdpgd, norm=float(norm), num_steps=num_steps, random_init=random_init,
                   proximal_operator=proximal_operator, primal_lr=primal_lr, primal_lr_decrease=primal_lr_decrease,
                   dual_ratio_init=dual_ratio_init, dual_lr=dual_lr, dual_lr_decrease=dual_lr_decrease,
                   dual_ema=dual_ema, dual_min_ratio=dual_min_ratio, proximal_steps=proximal_steps,
                   ε_threshold=ε_threshold)


def adv_lib_pgd():
    name = 'pgd'
    source = 'adv_lib'  # available: ['adv_lib']
    ε = 4 / 255
    steps = 40
    random_init = True
    restarts = 1
    loss_function = 'ce'
    relative_step_size = 0.01 / 0.3
    absolute_step_size = None


def get_adv_lib_pgd(ε: float, steps: int, random_init: bool, restarts: int, loss_function: str,
                    relative_step_size: float, absolute_step_size: Optional[float]):
    return partial(pgd_linf, ε=ε, steps=steps, random_init=random_init, restarts=restarts,
                   loss_function=loss_function, relative_step_size=relative_step_size,
                   absolute_step_size=absolute_step_size)


def adv_lib_tr():
    name = 'tr'
    source = 'adv_lib'  # available: ['adv_lib']
    p = float('inf')
    iter = 100
    adaptive = False
    eps = 0.001
    c = 9
    worst_case = False


def get_adv_lib_tr(p: float, iter: int, adaptive: bool, eps: float, c: int, worst_case: bool) -> Callable:
    return partial(tr, p=p, iter=iter, adaptive=adaptive, eps=eps, c=c, worst_case=worst_case)


def adv_lib_vfga():
    name = 'vfga'
    source = 'adv_lib'  # available: ['adv_lib']
    max_iter = None
    n_samples = 10
    large_memory = False


def get_adv_lib_vfga(max_iter: int, n_samples: int, large_memory: bool) -> Callable:
    return partial(vfga, max_iter=max_iter, n_samples=n_samples, large_memory=large_memory)


adv_lib_index = {
    'alma': ConfigGetter(config=adv_lib_alma, getter=get_adv_lib_alma),
    'apgd': ConfigGetter(config=adv_lib_apgd, getter=get_adv_lib_apgd),
    'cw_l2': ConfigGetter(config=adv_lib_cw_l2, getter=get_adv_lib_cw_l2),
    'cw_linf': ConfigGetter(config=adv_lib_cw_linf, getter=get_adv_lib_cw_linf),
    'ddn': ConfigGetter(config=adv_lib_ddn, getter=get_adv_lib_ddn),
    'fab': ConfigGetter(config=adv_lib_fab, getter=get_adv_lib_fab),
    'fmn': ConfigGetter(config=adv_lib_fmn, getter=get_adv_lib_fmn),
    'pdgd': ConfigGetter(config=adv_lib_pdgd, getter=get_adv_lib_pdgd),
    'pdpgd': ConfigGetter(config=adv_lib_pdpgd, getter=get_adv_lib_pdpgd),
    'pgd': ConfigGetter(config=adv_lib_pgd, getter=get_adv_lib_pgd),
    'tr': ConfigGetter(config=adv_lib_tr, getter=get_adv_lib_tr),
    'vfga': ConfigGetter(config=adv_lib_vfga, getter=get_adv_lib_vfga),
}
