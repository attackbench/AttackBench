from cmath import inf
from functools import partial
from typing import Callable, Optional
from torch import Tensor

from adv_lib.attacks import alma as alma_attack, ddn as ddn_attack, fmn as fmn_adv_lib_attack
from adv_lib.attacks import vfga as vfga_attack, pdgd as pdgd_attack
from .foolbox.foolbox_attacks import fb_dataset_attack
from .torchattacks.torch_attacks import sparsefool as sparsefool_attack

from sacred import Ingredient

from .adversarial_library import adv_lib_wrapper
from .original.fast_minimum_norm import fmn_attack
from .torchattacks.torch_attacks import torch_attacks_wrapper
from .art.art_attacks import art_lib_wrapper, art_lib_pgd, art_lib_fgsm, art_lib_jsma, art_lib_cw_l2, art_lib_cw_linf

attack_ingredient = Ingredient('attack')


@attack_ingredient.named_config
def alma():
    name = 'alma'
    origin = 'adv_lib'  # available: ['adv_lib']
    distance = 'l2'
    steps = 1000
    alpha = 0.9
    init_lr_distance = 1


@attack_ingredient.named_config
def ddn():
    name = 'ddn'
    origin = 'adv_lib'  # available: ['adv_lib']
    steps = 1000
    init_norm = 1
    gamma = 0.05


@attack_ingredient.named_config
def vfga():
    name = 'vfga'
    origin = 'adv_lib'
    targeted = False
    max_iter = None
    n_samples = 10
    large_memory = False


@attack_ingredient.named_config
def pdgd():
    name = 'pdgd'
    origin = 'adv_lib'
    targeted = False
    num_steps = 500
    random_init = 0
    primal_lr = 0.1
    primal_lr_decrease = 0.01
    dual_ratio_init = 0.01
    dual_lr = 0.1
    dual_lr_decrease = 0.1
    dual_ema = 0.9
    dual_min_ratio = 1e-6


@attack_ingredient.named_config
def fmn():
    name = 'fmn'
    origin = 'original'  # available: ['original', 'adv_lib']
    norm = 2
    steps = 1000
    max_stepsize = 1
    gamma = 0.05


@attack_ingredient.named_config
def dataset_attack():
    # Use default config from foolbox. By default pgd with l2 is executed.
    name = 'dataset_attack'
    origin = 'foolbox'  # available: ['foolbox']


@attack_ingredient.named_config
def sparsefool():
    name = 'sparsefool'
    origin = 'torchattacks'  # available: ['torchattacks']
    steps = 20
    lam = 3
    overshoot = 0.02


@attack_ingredient.named_config
def pgd():
    name = 'pgd'
    origin = 'art'  # available: ['art']
    norm = inf
    eps = 0.3
    eps_step = 0.1
    max_iter = 100
    num_random_init = 0
    random_eps = False


@attack_ingredient.named_config
def fgsm():
    name = 'fgsm'
    origin = 'art'  # available: ['art']
    norm = inf
    eps = 0.3
    eps_step = 0.1
    num_random_init = 0
    minimal = False


@attack_ingredient.named_config
def jsma():
    name = 'jsma'
    origin = 'art'  # available: ['art']
    theta = 0.1
    gamma = 1.0


@attack_ingredient.named_config
def cw_l2():
    name = 'cw_l2'
    origin = 'art'  # available: ['art']
    confidence = 0.0
    learning_rate = 0.01
    binary_search_steps = 10
    max_iter = 10
    initial_const = 0.01
    max_halving = 5
    max_doubling = 5


@attack_ingredient.named_config
def cw_linf():
    name = 'cw_linf'
    origin = 'art'  # available: ['art']
    confidence = 0.0
    learning_rate = 0.01
    max_iter = 10
    decrease_factor = 0.9
    initial_const = 0.01
    largest_const = 20.0
    const_factor = 2.0


@attack_ingredient.capture
def get_alma(distance: float, steps: int, alpha: float, init_lr_distance: float) -> Callable:
    return partial(alma_attack, distance=distance, num_steps=steps, α=alpha, init_lr_distance=init_lr_distance)


@attack_ingredient.capture
def get_ddn(steps: int, gamma: float, init_norm: float) -> Callable:
    return partial(ddn_attack, steps=steps, γ=gamma, init_norm=init_norm)


@attack_ingredient.capture
def get_fmn(norm: float, steps: int, max_stepsize: float, gamma: float) -> Callable:
    return partial(fmn_attack, norm=norm, steps=steps, max_stepsize=max_stepsize, gamma=gamma)


@attack_ingredient.capture
def get_adv_lib_fmn(norm: float, steps: int, max_stepsize: float, gamma: float) -> Callable:
    return partial(fmn_adv_lib_attack, norm=norm, steps=steps, α_init=max_stepsize, γ_init=gamma)


@attack_ingredient.capture
def get_adv_lib_vfga(targeted: bool, max_iter: int, n_samples: int, large_memory: bool) -> Callable:
    return partial(vfga_attack, targeted=targeted, max_iter=max_iter, n_samples=n_samples, large_memory=large_memory)


@attack_ingredient.capture
def get_adv_lib_pdgd(num_steps: int, random_init: float, primal_lr: float, primal_lr_decrease: float,
                     dual_ratio_init: float, dual_lr: float, dual_lr_decrease: float, dual_ema: float,
                     dual_min_ratio: float) -> Callable:
    return partial(pdgd_attack, num_steps=num_steps, random_init=random_init, primal_lr=primal_lr,
                   primal_lr_decrease=primal_lr_decrease, dual_ratio_init=dual_ratio_init, dual_lr=dual_lr,
                   dual_lr_decrease=dual_lr_decrease, dual_ema=dual_ema, dual_min_ratio=dual_min_ratio)


@attack_ingredient.capture
def get_dataset_attack() -> Callable:
    return fb_dataset_attack


@attack_ingredient.capture
def get_torchattacks_lib_sparsefool(steps: int, lam: float, overshoot: float) -> Callable:
    return partial(sparsefool_attack, steps=steps, lam=lam, overshoot=overshoot)


@attack_ingredient.capture
def get_art_lib_pgd(norm: float, eps: float, eps_step: float, max_iter: int,
                    num_random_init: int, random_eps: bool):
    return partial(art_lib_pgd, norm=norm, eps=eps, eps_step=eps_step,
                   num_random_init=num_random_init, max_iter=max_iter, random_eps=random_eps)


@attack_ingredient.capture
def get_art_lib_fgsm(norm: float, eps: float, eps_step: float,
                     num_random_init: int, minimal: bool):
    return partial(art_lib_fgsm, norm=norm, eps=eps, eps_step=eps_step,
                   num_random_init=num_random_init, minimal=minimal)


@attack_ingredient.capture
def get_art_lib_jsma(theta: float, gamma: float):
    return partial(art_lib_jsma, theta=theta, gamma=gamma)


@attack_ingredient.capture
def get_art_lib_cw_l2(confidence: float, learning_rate: float,
                      binary_search_steps: int, max_iter: int, initial_const: float,
                      max_halving: int, max_doubling: int):
    return partial(art_lib_cw_l2, confidence=confidence,
                   learning_rate=learning_rate, binary_search_steps=binary_search_steps,
                   max_iter=max_iter, initial_const=initial_const, max_halving=max_halving,
                   max_doubling=max_doubling)


@attack_ingredient.capture
def get_art_lib_cw_linf(confidence: float, learning_rate: float,
                        max_iter: int, decrease_factor: float, initial_const: float,
                        largest_const: float, const_factor: float):
    return partial(art_lib_cw_linf, confidence=confidence,
                   learning_rate=learning_rate, max_iter=max_iter,
                   decrease_factor=decrease_factor, initial_const=initial_const,
                   largest_const=largest_const,
                   const_factor=const_factor)


_original = {
    'fmn': get_fmn,
}


@attack_ingredient.capture
def get_original_attack(name: str) -> Callable:
    return _original[name]()


_adv_lib = {
    'alma': get_alma,
    'ddn': get_ddn,
    'fmn': get_adv_lib_fmn,
    'vfga': get_adv_lib_vfga,
    'pdgd': get_adv_lib_pdgd
}


@attack_ingredient.capture
def get_adv_lib_attack(name: str) -> Callable:
    attack = _adv_lib[name]()
    return partial(adv_lib_wrapper, attack=attack)


_foolbox_lib = {
    'dataset_attack': get_dataset_attack
}


@attack_ingredient.capture
def get_foolbox_lib_attack(name: str) -> Callable:
    return _foolbox_lib[name]()


_torchattacks_lib = {
    'sparsefool': get_torchattacks_lib_sparsefool
}

_art_lib = {
    'pgd': get_art_lib_pgd,
    'fgsm': get_art_lib_fgsm,
    'jsma': get_art_lib_jsma,
    'cw_l2': get_art_lib_cw_l2,
    'cw_linf': get_art_lib_cw_linf,
}


@attack_ingredient.capture
def get_torchattacks_lib_attack(name: str) -> Callable:
    attack = _torchattacks_lib[name]()
    return partial(torch_attacks_wrapper, attack=attack)


@attack_ingredient.capture
def get_art_lib_attack(name: str) -> Callable:
    attack = _art_lib[name]()
    return partial(art_lib_wrapper, attack=attack)


_libraries = {
    'original': get_original_attack,
    'adv_lib': get_adv_lib_attack,
    'foolbox': get_foolbox_lib_attack,
    'torchattacks': get_torchattacks_lib_attack,
    'art': get_art_lib_attack,
}


@attack_ingredient.capture
def get_attack(origin: str) -> Callable:
    return _libraries[origin]()
