from functools import partial
from typing import Callable, Optional

from sacred import Ingredient

from .adv_lib import adv_lib_index, adv_lib_wrapper
from .art import art_index, art_wrapper
from .deeprobust import deeprobust_index, deeprobust_wrapper
from .foolbox import foolbox_index, foolbox_wrapper
from .original import apgd_attack, apgd_t_attack, deepfool_attack, fab_attack, fmn_attack, tr_attack
from .torchattacks import torchattacks_index, torchattacks_wrapper
from .cleverhans import cleverhans_index, cleverhans_wrapper

attack_ingredient = Ingredient('attack')

for index in [adv_lib_index, art_index, deeprobust_index, foolbox_index, torchattacks_index, cleverhans_index]:
    for attack in index.values():
        attack_ingredient.named_config(attack.config)
        attack.getter = attack_ingredient.capture(attack.getter)


@attack_ingredient.named_config
def apgd():
    name = 'apgd'
    source = 'original'
    norm = float('inf')
    num_steps = 100
    num_restarts = 1
    epsilon = 0.3
    loss = 'ce'  # loss function in ['ce', 'dlr']
    rho = .75
    use_largereps = False  # set True with L1 norm


@attack_ingredient.capture
def get_apgd(norm: float, num_steps: int, num_restarts: int, epsilon: float, loss: str, rho: float,
             use_largereps: bool) -> Callable:
    return partial(apgd_attack, norm=norm, n_iter=num_steps, n_restarts=num_restarts, eps=epsilon, loss=loss, rho=rho,
                   use_largereps=use_largereps)


@attack_ingredient.named_config
def apgd_t():
    name = 'apgd_t'
    source = 'original'
    norm = float('inf')
    num_steps = 100
    num_restarts = 1
    num_target_classes = 9
    epsilon = 0.3
    rho = .75
    use_largereps = False  # set True with L1 norm


@attack_ingredient.capture
def get_apgd_t(norm: float, num_steps: int, num_restarts: int, num_target_classes: int, epsilon: float, rho: float,
               use_largereps: bool) -> Callable:
    return partial(apgd_t_attack, norm=norm, n_iter=num_steps, n_restarts=num_restarts,
                   n_target_classes=num_target_classes, eps=epsilon, rho=rho, use_largereps=use_largereps)


@attack_ingredient.named_config
def deepfool():
    name = 'deepfool'
    source = 'original'
    num_classes = 10  # number of classes to test gradient (can be different from the number of classes of the model)
    overshoot = 0.02
    num_steps = 50


@attack_ingredient.capture
def get_deepfool(num_classes: int, overshoot: float, num_steps: int) -> Callable:
    return partial(deepfool_attack, num_classes=num_classes, overshoot=overshoot, max_iter=num_steps)


@attack_ingredient.named_config
def fab():
    name = 'fab'
    source = 'original'
    norm = float('inf')
    num_restarts = 1
    num_steps = 100
    epsilon = None
    alpha_max = 0.1
    eta = 1.05
    beta = 0.9
    targeted_variant = False
    n_target_classes = 9


@attack_ingredient.capture
def get_fab(norm: float, num_restarts: int, num_steps: int, epsilon: Optional[float], alpha_max: float, eta: float,
            beta: float, targeted_variant: bool, n_target_classes: int) -> Callable:
    return partial(fab_attack, norm=norm, n_restarts=num_restarts, n_iter=num_steps, eps=epsilon, alpha_max=alpha_max,
                   eta=eta, beta=beta, targeted_variant=targeted_variant, n_target_classes=n_target_classes)


@attack_ingredient.named_config
def fmn():
    name = 'fmn'
    source = 'original'  # available: ['original', 'adv_lib']
    norm = 2
    num_steps = 1000
    max_step_size = 1
    gamma = 0.05


@attack_ingredient.capture
def get_fmn(norm: float, num_steps: int, max_step_size: float, gamma: float) -> Callable:
    return partial(fmn_attack, norm=norm, steps=num_steps, max_stepsize=max_step_size, gamma=gamma)


@attack_ingredient.named_config
def tr():
    name = 'tr'
    source = 'original'
    norm = float('inf')
    adaptive = False
    epsilon = 0.001
    c = 9
    num_steps = 100


@attack_ingredient.capture
def get_tr(norm: float, adaptive: bool, epsilon: float, c: int, num_steps: int) -> Callable:
    return partial(tr_attack, norm=norm, adaptive=adaptive, eps=epsilon, c=c, iter=num_steps)


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


@attack_ingredient.capture
def get_foolbox_attack(name: str) -> Callable:
    attack = foolbox_index[name].getter()
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


@attack_ingredient.capture
def get_cleverhans_attack(name: str) -> Callable:
    attack = cleverhans_index[name].getter()
    return partial(cleverhans_wrapper, attack=attack)


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
def get_attack(source: str) -> Callable:
    return _libraries[source]()
