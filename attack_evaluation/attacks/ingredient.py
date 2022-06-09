from cmath import inf
from functools import partial
from typing import Callable, Optional

from foolbox.attacks.base import MinimizationAttack
from sacred import Ingredient

from .adv_lib import adv_lib_index, adv_lib_wrapper
from .art import art_index, art_wrapper
from .deeprobust import deeprobust_index, deeprobust_wrapper
from .foolbox.foolbox_attacks import fb_lib_dataset_attack, fb_lib_fmn_attack, foolbox_wrapper
from .original.fast_minimum_norm import fmn_attack
from .torchattacks import torchattacks_index, torchattacks_wrapper

attack_ingredient = Ingredient('attack')

for index in [adv_lib_index, art_index, deeprobust_index, torchattacks_index]:
    for attack in index.values():
        attack_ingredient.named_config(attack.config)
        attack.getter = attack_ingredient.capture(attack.getter)


@attack_ingredient.named_config
def fmn():
    name = 'fmn'
    source = 'original'  # available: ['original', 'adv_lib']
    norm = 2
    steps = 1000
    max_stepsize = 1
    gamma = 0.05


@attack_ingredient.capture
def get_fmn(norm: float, steps: int, max_stepsize: float, gamma: float) -> Callable:
    return partial(fmn_attack, norm=norm, steps=steps, max_stepsize=max_stepsize, gamma=gamma)


@attack_ingredient.named_config
def fb_fmn():
    name = 'fmn'
    source = 'foolbox'  # available: ['foolbox', 'adv_lib', 'original']
    norm = inf
    steps = 100
    max_stepsize = 1.0
    min_stepsize = None
    gamma = 0.05
    init_attack = None
    binary_search_steps = 10


@attack_ingredient.capture
def get_fb_fmn(norm: float, steps: int, max_stepsize: float, gamma: float, min_stepsize: Optional[float],
               init_attack: Optional[MinimizationAttack], binary_search_steps: int) -> Callable:
    return partial(fb_lib_fmn_attack, norm=norm, steps=steps, max_stepsize=max_stepsize, min_stepsize=min_stepsize,
                   gamma=gamma, init_attack=init_attack, binary_search_steps=binary_search_steps)


@attack_ingredient.named_config
def fb_dataset_attack():
    # Use default config from foolbox. By default pgd with l2 is executed.
    name = 'dataset_attack'
    source = 'foolbox'  # available: ['foolbox']


@attack_ingredient.capture
def get_dataset_attack() -> Callable:
    return fb_lib_dataset_attack


_original = {
    'fmn': get_fmn,
}


@attack_ingredient.capture
def get_original_attack(name: str) -> Callable:
    return _original[name]()


@attack_ingredient.capture
def get_adv_lib_attack(name: str) -> Callable:
    attack = adv_lib_index[name].getter()
    return partial(adv_lib_wrapper, attack=attack)


_foolbox_lib = {
    'dataset_attack': get_dataset_attack,
    'fmn': get_fb_fmn
}


@attack_ingredient.capture
def get_foolbox_attack(name: str) -> Callable:
    attack = _foolbox_lib[name]()
    return partial(foolbox_wrapper, attack=attack)


@attack_ingredient.capture
def get_torchattacks_attack(name: str) -> Callable:
    attack = torchattacks_index[name].getter()
    return partial(torchattacks_wrapper, attack=attack)


@attack_ingredient.capture
def get_art_attack(name: str) -> Callable:
    attack = art_index[name].getter()
    return partial(art_wrapper, attack=attack)


@attack_ingredient.capture
def get_deeprobust_attack(name: str) -> Callable:
    attack, attack_params = deeprobust_index[name].getter()
    return partial(deeprobust_wrapper, attack=attack, attack_params=attack_params)


_libraries = {
    'original': get_original_attack,
    'adv_lib': get_adv_lib_attack,
    'art': get_art_attack,
    'deeprobust': get_deeprobust_attack,
    'foolbox': get_foolbox_attack,
    'torchattacks': get_torchattacks_attack,
}


@attack_ingredient.capture
def get_attack(source: str) -> Callable:
    return _libraries[source]()
