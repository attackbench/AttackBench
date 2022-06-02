from functools import partial
from typing import Callable, Optional

from adv_lib.attacks import (
    alma as alma_attack,
    ddn as ddn_attack,
    fmn as fmn_adv_lib_attack,
    pdgd as pdgd_attack,
    pdpgd as pdpgd_attack,
    vfga as vfga_attack
)


def adv_lib_alma():
    name = 'alma'
    source = 'adv_lib'  # available: ['adv_lib']
    distance = 'l2'
    steps = 1000
    alpha = 0.9
    init_lr_distance = 1


def get_adv_lib_alma(distance: float, steps: int, alpha: float, init_lr_distance: float) -> Callable:
    return partial(alma_attack, distance=distance, num_steps=steps, α=alpha, init_lr_distance=init_lr_distance)


def adv_lib_ddn():
    name = 'ddn'
    source = 'adv_lib'  # available: ['adv_lib']
    steps = 1000
    init_norm = 1
    gamma = 0.05


def get_adv_lib_ddn(steps: int, gamma: float, init_norm: float) -> Callable:
    return partial(ddn_attack, steps=steps, γ=gamma, init_norm=init_norm)


def adv_lib_vfga():
    name = 'vfga'
    source = 'adv_lib'  # available: ['adv_lib']
    max_iter = None
    n_samples = 10
    large_memory = False


def get_adv_lib_vfga(max_iter: int, n_samples: int, large_memory: bool) -> Callable:
    return partial(vfga_attack, max_iter=max_iter, n_samples=n_samples, large_memory=large_memory)


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
    return partial(pdgd_attack, num_steps=num_steps, random_init=random_init, primal_lr=primal_lr,
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
    return partial(pdpgd_attack, norm=float(norm), num_steps=num_steps, random_init=random_init,
                   proximal_operator=proximal_operator, primal_lr=primal_lr, primal_lr_decrease=primal_lr_decrease,
                   dual_ratio_init=dual_ratio_init, dual_lr=dual_lr, dual_lr_decrease=dual_lr_decrease,
                   dual_ema=dual_ema, dual_min_ratio=dual_min_ratio, proximal_steps=proximal_steps,
                   ε_threshold=ε_threshold)


def adv_lib_fmn():
    name = 'fmn'
    source = 'original'  # available: ['original', 'adv_lib']
    norm = 2
    steps = 1000
    max_stepsize = 1
    gamma = 0.05


def get_adv_lib_fmn(norm: float, steps: int, max_stepsize: float, gamma: float) -> Callable:
    return partial(fmn_adv_lib_attack, norm=norm, steps=steps, α_init=max_stepsize, γ_init=gamma)


adv_lib_index = {
    'alma': {'config': adv_lib_alma, 'getter': get_adv_lib_alma},
    'ddn': {'config': adv_lib_ddn, 'getter': get_adv_lib_ddn},
    'fmn': {'config': adv_lib_fmn, 'getter': get_adv_lib_fmn},
    'pdgd': {'config': adv_lib_pdgd, 'getter': get_adv_lib_pdgd},
    'pdpgd': {'config': adv_lib_pdpgd, 'getter': get_adv_lib_pdpgd},
    'vfga': {'config': adv_lib_vfga, 'getter': get_adv_lib_vfga},
}
