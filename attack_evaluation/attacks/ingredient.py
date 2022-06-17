from cmath import inf
from functools import partial
from typing import Callable, Optional

from foolbox.attacks.base import MinimizationAttack
from sacred import Ingredient

from .adv_lib import adv_lib_index, adv_lib_wrapper
from .art import art_index, art_wrapper
from .deeprobust import deeprobust_index, deeprobust_wrapper
from .foolbox.foolbox_attacks import fb_lib_dataset_attack, fb_lib_fmn_attack, foolbox_wrapper
from .original import apgd_attack, apgd_t_attack, deepfool_attack, fab_attack, fmn_attack, tr_attack
from .torchattacks import torchattacks_index, torchattacks_wrapper

attack_ingredient = Ingredient('attack')

for index in [adv_lib_index, art_index, deeprobust_index, torchattacks_index]:
    for attack in index.values():
        attack_ingredient.named_config(attack.config)
        attack.getter = attack_ingredient.capture(attack.getter)


@attack_ingredient.named_config
def apgd():
    name = 'apgd'
    source = 'original'
    norm = float('inf')
    n_iter = 100
    n_restarts = 1
    eps = 0.3
    loss = 'ce'  # loss function in ['ce', 'dlr']
    rho = .75
    use_largereps = False  # set True with L1 norm


@attack_ingredient.capture
def get_apgd(norm: float, n_iter: int, n_restarts: int, eps: float, loss: str, rho: float,
             use_largereps: bool) -> Callable:
    return partial(apgd_attack, norm=norm, n_iter=n_iter, n_restarts=n_restarts, eps=eps, loss=loss, rho=rho,
                   use_largereps=use_largereps)


@attack_ingredient.named_config
def apgd_t():
    name = 'apgd_t'
    source = 'original'
    norm = float('inf')
    n_iter = 100
    n_restarts = 1
    n_target_classes = 9
    eps = 0.3
    rho = .75
    use_largereps = False  # set True with L1 norm


@attack_ingredient.capture
def get_apgd_t(norm: float, n_iter: int, n_restarts: int, n_target_classes: int, eps: float, rho: float,
               use_largereps: bool) -> Callable:
    return partial(apgd_t_attack, norm=norm, n_iter=n_iter, n_restarts=n_restarts, n_target_classes=n_target_classes,
                   eps=eps, rho=rho, use_largereps=use_largereps)


@attack_ingredient.named_config
def deepfool():
    name = 'deepfool'
    source = 'original'
    num_classes = 10  # number of classes to test gradient (can be different from the number of classes of the model)
    overshoot = 0.02
    max_iter = 50


@attack_ingredient.capture
def get_deepfool(num_classes: int, overshoot: float, max_iter: int) -> Callable:
    return partial(deepfool_attack, num_classes=num_classes, overshoot=overshoot, max_iter=max_iter)


@attack_ingredient.named_config
def fab():
    name = 'fab'
    source = 'original'
    norm = float('inf')
    n_restarts = 1
    n_iter = 100
    eps = None
    alpha_max = 0.1
    eta = 1.05
    beta = 0.9
    targeted_variant = False
    n_target_classes = 9


@attack_ingredient.capture
def get_fab(norm: float, n_restarts: int, n_iter: int, eps: Optional[float], alpha_max: float, eta: float, beta: float,
            targeted_variant: bool, n_target_classes: int) -> Callable:
    return partial(fab_attack, norm=norm, n_restarts=n_restarts, n_iter=n_iter, eps=eps, alpha_max=alpha_max, eta=eta,
                   beta=beta, targeted_variant=targeted_variant, n_target_classes=n_target_classes)


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
def tr():
    name = 'tr'
    source = 'original'
    norm = float('inf')
    adaptive = False
    eps = 0.001
    c = 9
    iter = 100


@attack_ingredient.capture
def get_tr(norm: float, adaptive: bool, eps: float, c: int, iter: int) -> Callable:
    return partial(tr_attack, norm=norm, adaptive=adaptive, eps=eps, c=c, iter=iter)


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
    'apgd': get_apgd,
    'apgd_t': get_apgd_t,
    'deepfool': get_deepfool,
    'fab': get_fab,
    'fmn': get_fmn,
    'tr': get_tr,
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
