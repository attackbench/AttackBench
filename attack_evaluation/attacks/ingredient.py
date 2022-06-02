from cmath import inf
from functools import partial
from typing import Callable, Optional

from foolbox.attacks.base import MinimizationAttack
from sacred import Ingredient

from .adv_lib import adv_lib_index, adv_lib_wrapper
from .art.art_attacks import (
    art_lib_apgd,
    art_lib_bb,
    art_lib_bim,
    art_lib_cw_l2,
    art_lib_cw_linf,
    art_lib_deepfool,
    art_lib_ead,
    art_lib_fgsm,
    art_lib_jsma,
    art_lib_pgd,
    art_lib_wrapper
)
from .foolbox.foolbox_attacks import fb_lib_dataset_attack, fb_lib_fmn_attack, foolbox_wrapper
from .original.fast_minimum_norm import fmn_attack
from .torchattacks.torch_attacks import (
    ta_lib_auto_attack,
    ta_lib_cw,
    ta_lib_deepfool,
    ta_lib_fab,
    ta_lib_fgsm,
    ta_lib_pgd_l2,
    ta_lib_pgd_linf,
    ta_lib_sparsefool,
    torch_attacks_wrapper
)

attack_ingredient = Ingredient('attack')

for attack in adv_lib_index.values():
    attack_ingredient.named_config(attack['config'])
    attack['getter'] = attack_ingredient.capture(attack['getter'])


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


@attack_ingredient.named_config
def ta_deepfool():
    name = 'ta_deepfool'
    source = 'torchattacks'  # available: ['torchattacks', 'art']
    norm = 2
    steps = 50
    overshoot = 0.02


@attack_ingredient.capture
def get_torchattacks_lib_deepfool(norm: float, steps: int, overshoot: float) -> Callable:
    return partial(ta_lib_deepfool, steps=steps, norm=norm, overshoot=overshoot)


@attack_ingredient.named_config
def ta_sparsefool():
    name = 'sparsefool'
    source = 'torchattacks'  # available: ['torchattacks']
    norm = 0
    steps = 20
    lam = 3
    overshoot = 0.02


@attack_ingredient.capture
def get_torchattacks_lib_sparsefool(norm: float, steps: int, lam: float, overshoot: float) -> Callable:
    return partial(ta_lib_sparsefool, norm=norm, steps=steps, lam=lam, overshoot=overshoot)


@attack_ingredient.named_config
def ta_fab():
    name = 'fab'
    source = 'torchattacks'  # available: ['torchattacks']
    norm = 2  # available: inf, 2, 1
    steps = 100
    eps = None
    n_restarts = 1
    alpha_max = 0.1
    eta = 1.05
    beta = 0.9
    targeted = False


@attack_ingredient.capture
def get_torchattacks_lib_fab(norm: float, eps: float, steps: int, n_restarts: int, alpha_max: float, eta: float,
                             beta: float, targeted: bool) -> Callable:
    return partial(ta_lib_fab, steps=steps, norm=norm, eps=eps, n_restarts=n_restarts, alpha_max=alpha_max, eta=eta,
                   beta=beta, targeted=targeted)


@attack_ingredient.named_config
def ta_fgsm():
    name = 'fgsm'
    source = 'torchattacks'  # available: ['torchattacks']
    norm = 2
    eps = 0.007


@attack_ingredient.capture
def get_torchattacks_lib_fgsm(norm: float, steps: int) -> Callable:
    return partial(ta_lib_fgsm, norm=norm, steps=steps)


@attack_ingredient.named_config
def ta_cw():
    name = 'cw'
    source = 'torchattacks'  # available: ['torchattacks']
    norm = 2
    c = 0.0001
    steps = 1000
    kappa = 0
    lr = 0.01


@attack_ingredient.capture
def get_torchattacks_lib_cw(norm: float, steps: int, c: float, kappa: float, lr: float) -> Callable:
    return partial(ta_lib_cw, norm=norm, steps=steps, c=c, kappa=kappa, lr=lr)


@attack_ingredient.named_config
def ta_pgd_linf():
    name = 'pgd_linf'
    source = 'torchattacks'  # available: ['torchattacks']
    norm = inf
    eps = 0.3
    alpha = 0.00784313725490196
    steps = 40
    random_start = True


@attack_ingredient.capture
def get_torchattacks_lib_pgd_linf(norm: float, steps: int, eps: float, alpha: float, random_start: bool) -> Callable:
    return partial(ta_lib_pgd_linf, norm=norm, steps=steps, eps=eps, alpha=alpha, random_start=random_start)


@attack_ingredient.named_config
def ta_pgd_l2():
    name = 'pgd_l2'
    source = 'torchattacks'  # available: ['torchattacks']
    norm = 2
    eps = 1.0
    alpha = 0.2
    steps = 40
    random_start = True
    eps_for_division = 1e-10


@attack_ingredient.capture
def get_torchattacks_lib_pgd_l2(norm: float, steps: int, eps: float, alpha: float, random_start: bool,
                                eps_for_division: float) -> Callable:
    return partial(ta_lib_pgd_l2, norm=norm, steps=steps, eps=eps, alpha=alpha, random_start=random_start,
                   eps_for_division=eps_for_division)


@attack_ingredient.named_config
def ta_auto_attack():
    name = 'auto_attack'
    source = 'torchattacks'  # available: ['torchattacks']
    norm = 2  # available: inf, 2
    eps = 0.3
    version = 'standard'


@attack_ingredient.capture
def get_torchattacks_lib_auto_attack(norm: float, eps: float, version: str) -> Callable:
    return partial(ta_lib_auto_attack, norm=norm, eps=eps, version=version)


@attack_ingredient.named_config
def pgd():
    name = 'pgd'
    source = 'art'  # available: ['art']
    norm = inf
    eps = 0.3
    eps_step = 0.1
    max_iter = 100
    num_random_init = 0
    random_eps = False


@attack_ingredient.capture
def get_art_lib_pgd(norm: float, eps: float, eps_step: float, max_iter: int, num_random_init: int,
                    random_eps: bool) -> Callable:
    return partial(art_lib_pgd, norm=norm, eps=eps, eps_step=eps_step, num_random_init=num_random_init,
                   max_iter=max_iter, random_eps=random_eps)


@attack_ingredient.named_config
def fgsm():
    name = 'fgsm'
    source = 'art'  # available: ['art']
    norm = inf
    eps = 0.3
    eps_step = 0.1
    num_random_init = 0
    minimal = False


@attack_ingredient.capture
def get_art_lib_fgsm(norm: float, eps: float, eps_step: float, num_random_init: int, minimal: bool) -> Callable:
    return partial(art_lib_fgsm, norm=norm, eps=eps, eps_step=eps_step, num_random_init=num_random_init,
                   minimal=minimal)


@attack_ingredient.named_config
def jsma():
    name = 'jsma'
    source = 'art'  # available: ['art']
    theta = 0.1
    gamma = 1.0


@attack_ingredient.capture
def get_art_lib_jsma(theta: float, gamma: float) -> Callable:
    return partial(art_lib_jsma, theta=theta, gamma=gamma)


@attack_ingredient.named_config
def cw_l2():
    name = 'cw_l2'
    source = 'art'  # available: ['art']
    confidence = 0.0
    learning_rate = 0.01
    binary_search_steps = 10
    max_iter = 10
    initial_const = 0.01
    max_halving = 5
    max_doubling = 5


@attack_ingredient.capture
def get_art_lib_cw_l2(confidence: float, learning_rate: float, binary_search_steps: int, max_iter: int,
                      initial_const: float, max_halving: int, max_doubling: int) -> Callable:
    return partial(art_lib_cw_l2, confidence=confidence, learning_rate=learning_rate,
                   binary_search_steps=binary_search_steps, max_iter=max_iter, initial_const=initial_const,
                   max_halving=max_halving, max_doubling=max_doubling)


@attack_ingredient.named_config
def cw_linf():
    name = 'cw_linf'
    source = 'art'  # available: ['art']
    confidence = 0.0
    learning_rate = 0.01
    max_iter = 10
    decrease_factor = 0.9
    initial_const = 0.01
    largest_const = 20.0
    const_factor = 2.0


@attack_ingredient.capture
def get_art_lib_cw_linf(confidence: float, learning_rate: float, max_iter: int, decrease_factor: float,
                        initial_const: float, largest_const: float, const_factor: float) -> Callable:
    return partial(art_lib_cw_linf, confidence=confidence, learning_rate=learning_rate, max_iter=max_iter,
                   decrease_factor=decrease_factor, initial_const=initial_const, largest_const=largest_const,
                   const_factor=const_factor)


@attack_ingredient.named_config
def bb():
    name = 'bb'
    source = 'art'  # available: ['art']
    norm = inf
    overshoot = 1.1
    steps = 1000
    lr = 1e-3
    lr_decay = 0.5
    lr_num_decay = 20
    momentum = 0.8
    binary_search_steps = 10
    init_size = 32


@attack_ingredient.capture
def get_art_lib_bb(norm: float, overshoot: float, steps: int, lr: float, lr_decay: float, lr_num_decay: int,
                   momentum: float, binary_search_steps: int, init_size: int) -> Callable:
    return partial(art_lib_bb, norm=norm, overshoot=overshoot, steps=steps, lr=lr, lr_decay=lr_decay,
                   lr_num_decay=lr_num_decay, momentum=momentum, binary_search_steps=binary_search_steps,
                   init_size=init_size)


@attack_ingredient.named_config
def deepfool():
    name = 'deepfool'
    source = 'art'  # available: ['art']
    max_iter = 100
    epsilon = 1e-6
    nb_grads = 10


@attack_ingredient.capture
def get_art_lib_deepfool(max_iter: int, epsilon: float, nb_grads: int) -> Callable:
    return partial(art_lib_deepfool, max_iter=max_iter, epsilon=epsilon, nb_grads=nb_grads)


@attack_ingredient.named_config
def apgd():
    name = 'apgd'
    source = 'art'  # available: ['art']
    norm = inf
    eps = 0.3
    eps_step = 0.1
    max_iter = 100
    nb_random_init = 5
    loss_type = None


@attack_ingredient.capture
def get_art_lib_apgd(norm: float, eps: float, eps_step: float, max_iter: int, nb_random_init: int,
                     loss_type: Optional[str]) -> Callable:
    return partial(art_lib_apgd, norm=norm, eps=eps, eps_step=eps_step, max_iter=max_iter,
                   nb_random_init=nb_random_init, loss_type=loss_type)


@attack_ingredient.named_config
def bim():
    name = 'bim'
    source = 'art'  # available: ['art']
    eps = 0.3
    eps_step = 0.1
    max_iter = 100


@attack_ingredient.capture
def get_art_lib_bim(eps: float, eps_step: float, max_iter: int) -> Callable:
    return partial(art_lib_bim, eps=eps, eps_step=eps_step, max_iter=max_iter)


@attack_ingredient.named_config
def ead():
    name = 'ead'
    source = 'art'  # available: ['art']
    confidence = 0.0
    learning_rate = 1e-2
    binary_search_steps = 9
    max_iter = 100
    beta = 1e-3
    initial_const = 1e-3
    decision_rule = 'EN'


@attack_ingredient.capture
def get_art_lib_ead(confidence: float, learning_rate: float, binary_search_steps: int, max_iter: int, beta: float,
                    initial_const: float, decision_rule: str) -> Callable:
    return partial(art_lib_ead, confidence=confidence, learning_rate=learning_rate,
                   binary_search_steps=binary_search_steps, max_iter=max_iter, beta=beta,
                   initial_const=initial_const, decision_rule=decision_rule)


_original = {
    'fmn': get_fmn,
}


@attack_ingredient.capture
def get_original_attack(name: str) -> Callable:
    return _original[name]()


@attack_ingredient.capture
def get_adv_lib_attack(name: str) -> Callable:
    attack = adv_lib_index[name]['getter']()
    return partial(adv_lib_wrapper, attack=attack)


_foolbox_lib = {
    'dataset_attack': get_dataset_attack,
    'fmn': get_fb_fmn
}


@attack_ingredient.capture
def get_foolbox_lib_attack(name: str) -> Callable:
    attack = _foolbox_lib[name]()
    return partial(foolbox_wrapper, attack=attack)


_torchattacks_lib = {
    'deepfool': get_torchattacks_lib_deepfool,
    'sparsefool': get_torchattacks_lib_sparsefool,
    'fab': get_torchattacks_lib_fab,
    'cw': get_torchattacks_lib_cw,
    'auto_attack': get_torchattacks_lib_auto_attack,
    'pgd_l2': get_torchattacks_lib_pgd_l2,
    'pgd_linf': get_torchattacks_lib_pgd_linf,
    'fgsm': get_torchattacks_lib_fgsm,
}

_art_lib = {
    'pgd': get_art_lib_pgd,
    'fgsm': get_art_lib_fgsm,
    'jsma': get_art_lib_jsma,
    'cw_l2': get_art_lib_cw_l2,
    'cw_linf': get_art_lib_cw_linf,
    'bb': get_art_lib_bb,
    'deepfool': get_art_lib_deepfool,
    'apgd': get_art_lib_apgd,
    'bim': get_art_lib_bim,
    'ead': get_art_lib_ead,
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
def get_attack(source: str) -> Callable:
    return _libraries[source]()
