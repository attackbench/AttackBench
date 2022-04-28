from functools import partial
from typing import Callable

from adv_lib.attacks import alma as alma_attack, ddn as ddn_attack, fmn as fmn_adv_lib_attack
from sacred import Ingredient

from .adversarial_library import adv_lib_wrapper
from .original.fast_minimum_norm import fmn_attack

attack_ingredient = Ingredient('attack')


@attack_ingredient.named_config
def alma():
    name = 'alma'
    origin = 'adv_lib'
    distance = 'l2'
    steps = 1000
    alpha = 0.9
    init_lr_distance = 1


@attack_ingredient.named_config
def ddn():
    name = 'ddn'
    origin = 'adv_lib'
    steps = 1000
    init_norm = 1
    gamma = 0.05


@attack_ingredient.named_config
def fmn():
    name = 'fmn'
    origin = 'original'
    norm = 2
    steps = 1000
    max_stepsize = 1
    gamma = 0.05


@attack_ingredient.capture
def get_alma(norm: float, num_steps: int, alpha: float, init_lr_distance: float) -> Callable:
    return partial(alma_attack, norm=norm, num_steps=num_steps, α=alpha, init_lr_distance=init_lr_distance)


@attack_ingredient.capture
def get_ddn(steps: int, gamma: float, init_norm: float) -> Callable:
    return partial(ddn_attack, steps=steps, γ=gamma, init_norm=init_norm)


@attack_ingredient.capture
def get_fmn(norm: float, steps: int, max_stepsize: float, gamma: float) -> Callable:
    return partial(fmn_attack, norm=norm, steps=steps, max_stepsize=max_stepsize, gamma=gamma)


@attack_ingredient.capture
def get_adv_lib_fmn(norm: float, steps: int, max_stepsize: float, gamma: float) -> Callable:
    return partial(fmn_adv_lib_attack, norm=norm, steps=steps, α_init=max_stepsize, γ_init=gamma)


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
}


@attack_ingredient.capture
def get_adv_lib_attack(name: str) -> Callable:
    attack = _adv_lib[name]()
    return partial(adv_lib_wrapper, attack=attack)


_libraries = {
    'original': get_original_attack,
    'adv_lib': get_adv_lib_attack,
}


@attack_ingredient.capture
def get_attack(origin: str) -> Callable:
    return _libraries[origin]()
