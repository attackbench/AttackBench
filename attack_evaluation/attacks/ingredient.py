import itertools
from functools import partial
from typing import Callable, Optional

from sacred import Ingredient

from .adv_lib import adv_lib_configs, adv_lib_getters, adv_lib_wrapper
from .art import art_configs, art_getters, art_wrapper
from .cleverhans import cleverhans_configs, cleverhans_getters, cleverhans_wrapper
from .deeprobust import deeprobust_configs, deeprobust_getters, deeprobust_wrapper
from .foolbox import foolbox_configs, foolbox_getters, foolbox_wrapper
from .original import apgd_attack, apgd_t_attack, deepfool_attack, fab_attack, fmn_attack, tr_attack
from .torchattacks import torchattacks_configs, torchattacks_getters, torchattacks_wrapper

attack_ingredient = Ingredient('attack')

configs = [adv_lib_configs, art_configs, cleverhans_configs, deeprobust_configs, foolbox_configs, torchattacks_configs]
getters = [adv_lib_getters, art_getters, cleverhans_getters, deeprobust_getters, foolbox_getters, torchattacks_getters]

for config in itertools.chain.from_iterable(configs):
    attack_ingredient.named_config(config)

for getter in getters:
    for attack in getter.keys():
        getter[attack] = attack_ingredient.capture(getter[attack])


@attack_ingredient.named_config
def apgd():
    name = 'apgd'
    source = 'original'
    threat_model = 'linf'
    num_steps = 100
    num_restarts = 1
    epsilon = 0.3
    loss = 'ce'  # loss function in ['ce', 'dlr']
    rho = .75
    use_largereps = False  # set True with L1 norm


@attack_ingredient.capture
def get_apgd(threat_model: str, num_steps: int, num_restarts: int, epsilon: float, loss: str, rho: float,
             use_largereps: bool) -> Callable:
    return partial(apgd_attack, threat_model=threat_model, n_iter=num_steps, n_restarts=num_restarts, eps=epsilon,
                   loss=loss, rho=rho, use_largereps=use_largereps)


@attack_ingredient.named_config
def apgd_t():
    name = 'apgd_t'
    source = 'original'
    threat_model = 'linf'
    num_steps = 100
    num_restarts = 1
    num_target_classes = 9
    epsilon = 0.3
    rho = .75
    use_largereps = False  # set True with L1 norm


@attack_ingredient.capture
def get_apgd_t(threat_model: str, num_steps: int, num_restarts: int, num_target_classes: int, epsilon: float,
               rho: float, use_largereps: bool) -> Callable:
    return partial(apgd_t_attack, threat_model=threat_model, n_iter=num_steps, n_restarts=num_restarts,
                   n_target_classes=num_target_classes, eps=epsilon, rho=rho, use_largereps=use_largereps)


@attack_ingredient.named_config
def deepfool():
    name = 'deepfool'
    source = 'original'
    threat_model = 'l2'
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
    threat_model = 'linf'
    num_restarts = 1
    num_steps = 100
    epsilon = None
    alpha_max = 0.1
    eta = 1.05
    beta = 0.9
    targeted_variant = False
    n_target_classes = 9


@attack_ingredient.capture
def get_fab(threat_model: str, num_restarts: int, num_steps: int, epsilon: Optional[float], alpha_max: float,
            eta: float, beta: float, targeted_variant: bool, n_target_classes: int) -> Callable:
    return partial(fab_attack, threat_model=threat_model, n_restarts=num_restarts, n_iter=num_steps, eps=epsilon,
                   alpha_max=alpha_max, eta=eta, beta=beta, targeted_variant=targeted_variant,
                   n_target_classes=n_target_classes)


@attack_ingredient.named_config
def fmn():
    name = 'fmn'
    source = 'original'  # available: ['original', 'adv_lib']
    threat_model = 'linf'
    num_steps = 1000
    max_step_size = 1
    gamma = 0.05


@attack_ingredient.capture
def get_fmn(threat_model: str, num_steps: int, max_step_size: float, gamma: float) -> Callable:
    return partial(fmn_attack, threat_model=threat_model, steps=num_steps, max_stepsize=max_step_size, gamma=gamma)


@attack_ingredient.named_config
def tr():
    name = 'tr'
    source = 'original'
    threat_model = 'linf'
    adaptive = False
    epsilon = 0.001
    c = 9
    num_steps = 100


@attack_ingredient.capture
def get_tr(threat_model: str, adaptive: bool, epsilon: float, c: int, num_steps: int) -> Callable:
    return partial(tr_attack, threat_model=threat_model, adaptive=adaptive, eps=epsilon, c=c, iter=num_steps)


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
