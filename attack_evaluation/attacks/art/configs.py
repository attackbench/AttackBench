from functools import partial
from typing import Callable, Optional

from art.attacks.evasion import (
    AutoProjectedGradientDescent,
    BasicIterativeMethod,
    BrendelBethgeAttack,
    CarliniL2Method,
    CarliniLInfMethod,
    DeepFool,
    ElasticNet,
    FastGradientMethod,
    ProjectedGradientDescent,
    SaliencyMapMethod
)

from ..utils import ConfigGetter


def art_apgd():
    name = 'apgd'
    source = 'art'  # available: ['art']
    norm = float('inf')
    eps = 0.3
    eps_step = 0.1
    max_iter = 100
    nb_random_init = 5
    loss_type = None


def get_art_apgd(norm: float, eps: float, eps_step: float, max_iter: int, nb_random_init: int,
                 loss_type: Optional[str]) -> Callable:
    return partial(AutoProjectedGradientDescent, norm=norm, eps=eps, eps_step=eps_step, max_iter=max_iter,
                   nb_random_init=nb_random_init, loss_type=loss_type)


def art_bim():
    name = 'bim'
    source = 'art'  # available: ['art']
    eps = 0.3
    eps_step = 0.1
    max_iter = 100


def get_art_bim(eps: float, eps_step: float, max_iter: int) -> Callable:
    return partial(BasicIterativeMethod, eps=eps, eps_step=eps_step, max_iter=max_iter)


def art_bb():
    name = 'bb'
    source = 'art'  # available: ['art']
    norm = float('inf')
    overshoot = 1.1
    steps = 1000
    lr = 1e-3
    lr_decay = 0.5
    lr_num_decay = 20
    momentum = 0.8
    binary_search_steps = 10
    init_size = 32


def get_art_bb(norm: float, overshoot: float, steps: int, lr: float, lr_decay: float, lr_num_decay: int,
               momentum: float, binary_search_steps: int, init_size: int) -> Callable:
    return partial(BrendelBethgeAttack, norm=norm, overshoot=overshoot, steps=steps, lr=lr, lr_decay=lr_decay,
                   lr_num_decay=lr_num_decay, momentum=momentum, binary_search_steps=binary_search_steps,
                   init_size=init_size)


def art_cw_l2():
    name = 'cw_l2'
    source = 'art'  # available: ['art']
    confidence = 0.0
    learning_rate = 0.01
    binary_search_steps = 10
    max_iter = 10
    initial_const = 0.01
    max_halving = 5
    max_doubling = 5


def get_art_cw_l2(confidence: float, learning_rate: float, binary_search_steps: int, max_iter: int,
                  initial_const: float, max_halving: int, max_doubling: int) -> Callable:
    return partial(CarliniL2Method, confidence=confidence, learning_rate=learning_rate,
                   binary_search_steps=binary_search_steps, max_iter=max_iter, initial_const=initial_const,
                   max_halving=max_halving, max_doubling=max_doubling)


def art_cw_linf():
    name = 'cw_linf'
    source = 'art'  # available: ['art']
    confidence = 0.0
    learning_rate = 0.01
    max_iter = 10
    decrease_factor = 0.9
    initial_const = 0.01
    largest_const = 20.0
    const_factor = 2.0


def get_art_cw_linf(confidence: float, learning_rate: float, max_iter: int, decrease_factor: float,
                    initial_const: float, largest_const: float, const_factor: float) -> Callable:
    return partial(CarliniLInfMethod, confidence=confidence, learning_rate=learning_rate, max_iter=max_iter,
                   decrease_factor=decrease_factor, initial_const=initial_const, largest_const=largest_const,
                   const_factor=const_factor)


def art_deepfool():
    name = 'deepfool'
    source = 'art'  # available: ['art']
    max_iter = 100
    epsilon = 1e-6
    nb_grads = 10


def get_art_deepfool(max_iter: int, epsilon: float, nb_grads: int) -> Callable:
    return partial(DeepFool, max_iter=max_iter, epsilon=epsilon, nb_grads=nb_grads)


def art_ead():
    name = 'ead'
    source = 'art'  # available: ['art']
    confidence = 0.0
    learning_rate = 1e-2
    binary_search_steps = 9
    max_iter = 100
    beta = 1e-3
    initial_const = 1e-3
    decision_rule = 'EN'


def get_art_ead(confidence: float, learning_rate: float, binary_search_steps: int, max_iter: int, beta: float,
                initial_const: float, decision_rule: str) -> Callable:
    return partial(ElasticNet, confidence=confidence, learning_rate=learning_rate,
                   binary_search_steps=binary_search_steps, max_iter=max_iter, beta=beta,
                   initial_const=initial_const, decision_rule=decision_rule)


def art_fgsm():
    name = 'fgsm'
    source = 'art'  # available: ['art']
    norm = float('inf')
    eps = 0.3
    eps_step = 0.1
    num_random_init = 0
    minimal = False


def get_art_fgsm(norm: float, eps: float, eps_step: float, num_random_init: int, minimal: bool) -> Callable:
    return partial(FastGradientMethod, norm=norm, eps=eps, eps_step=eps_step, num_random_init=num_random_init,
                   minimal=minimal)


def art_jsma():
    name = 'jsma'
    source = 'art'  # available: ['art']
    theta = 0.1
    gamma = 1.0


def get_art_jsma(theta: float, gamma: float) -> Callable:
    return partial(SaliencyMapMethod, theta=theta, gamma=gamma)


def art_pgd():
    name = 'pgd'
    source = 'art'  # available: ['art']
    norm = float('inf')
    eps = 0.3
    eps_step = 0.1
    max_iter = 100
    num_random_init = 0
    random_eps = False


def get_art_pgd(norm: float, eps: float, eps_step: float, max_iter: int, num_random_init: int,
                random_eps: bool) -> Callable:
    return partial(ProjectedGradientDescent, norm=norm, eps=eps, eps_step=eps_step, num_random_init=num_random_init,
                   max_iter=max_iter, random_eps=random_eps)


art_index = {
    'apgd': ConfigGetter(config=art_apgd, getter=get_art_apgd),
    'bim': ConfigGetter(config=art_bim, getter=get_art_bim),
    'bb': ConfigGetter(config=art_bb, getter=get_art_bb),
    'cw_l2': ConfigGetter(config=art_cw_l2, getter=get_art_cw_l2),
    'cw_linf': ConfigGetter(config=art_cw_linf, getter=get_art_cw_linf),
    'deepfool': ConfigGetter(config=art_deepfool, getter=get_art_deepfool),
    'ead': ConfigGetter(config=art_ead, getter=get_art_ead),
    'fgsm': ConfigGetter(config=art_fgsm, getter=get_art_fgsm),
    'jsma': ConfigGetter(config=art_jsma, getter=get_art_jsma),
    'pgd': ConfigGetter(config=art_pgd, getter=get_art_pgd),
}
