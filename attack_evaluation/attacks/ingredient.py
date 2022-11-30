import itertools
from functools import partial
from typing import Callable

from sacred import Ingredient

from .adv_lib import adv_lib_configs, adv_lib_getters, adv_lib_wrapper
from .art import art_configs, art_getters, art_wrapper
from .cleverhans import cleverhans_configs, cleverhans_getters, cleverhans_wrapper
from .deeprobust import deeprobust_configs, deeprobust_getters, deeprobust_wrapper
from .foolbox import foolbox_configs, foolbox_getters, foolbox_wrapper
from .original import original_configs, original_getters
from .torchattacks import torchattacks_configs, torchattacks_getters, torchattacks_wrapper

attack_ingredient = Ingredient('attack')

configs = [adv_lib_configs, art_configs, cleverhans_configs, deeprobust_configs, foolbox_configs, original_configs,
           torchattacks_configs]
getters = [adv_lib_getters, art_getters, cleverhans_getters, deeprobust_getters, foolbox_getters, original_getters,
           torchattacks_getters]

for config in itertools.chain.from_iterable(configs):
    attack_ingredient.named_config(config)

for getter in getters:
    for attack in getter.keys():
        getter[attack] = attack_ingredient.capture(getter[attack])


@attack_ingredient.capture
def get_original_attack(name: str) -> Callable:
    return original_getters[name]()


@attack_ingredient.capture
def get_adv_lib_attack(name: str) -> Callable:
    attack = adv_lib_getters[name]()
    return partial(adv_lib_wrapper, attack=attack)


@attack_ingredient.capture
def get_art_attack(name: str) -> Callable:
    attack = art_getters[name]()
    return partial(art_wrapper, attack=attack)


@attack_ingredient.capture
def get_cleverhans_attack(name: str) -> Callable:
    attack = cleverhans_getters[name]()
    return partial(cleverhans_wrapper, attack=attack)


@attack_ingredient.capture
def get_deeprobust_attack(name: str) -> Callable:
    attack, attack_params = deeprobust_getters[name]()
    return partial(deeprobust_wrapper, attack=attack, attack_params=attack_params)


@attack_ingredient.capture
def get_foolbox_attack(name: str) -> Callable:
    attack = foolbox_getters[name]()
    return partial(foolbox_wrapper, attack=attack)


@attack_ingredient.capture
def get_torchattacks_attack(name: str) -> Callable:
    attack = torchattacks_getters[name]()
    return partial(torchattacks_wrapper, attack=attack)


_libraries = {
    'original': get_original_attack,
    'adv_lib': get_adv_lib_attack,
    'art': get_art_attack,
    'deeprobust': get_deeprobust_attack,
    'foolbox': get_foolbox_attack,
    'torchattacks': get_torchattacks_attack,
    'cleverhans': get_cleverhans_attack
}


@attack_ingredient.capture
def get_attack(source: str, threat_model: str) -> Callable:
    return _libraries[source]()
