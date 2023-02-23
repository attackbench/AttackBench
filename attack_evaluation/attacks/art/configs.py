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

from .wrapper import ArtMinimalWrapper, art_wrapper

_prefix = 'art'
_wrapper = art_wrapper
_norms = {
    'l0': 0,
    'l1': 1,
    'l2': 2,
    'linf': float('inf'),
}


def art_apgd():
    name = 'apgd'
    source = 'art'
    threat_model = 'linf'
    epsilon = 0.3
    step_size = 0.1
    num_steps = 100
    nb_random_init = 5
    loss_type = None


def get_art_apgd(threat_model: str, epsilon: float, step_size: float, num_steps: int, nb_random_init: int,
                 loss_type: Optional[str]) -> Callable:
    return partial(AutoProjectedGradientDescent, norm=_norms[threat_model], eps=epsilon, eps_step=step_size,
                   max_iter=num_steps, nb_random_init=nb_random_init, loss_type=loss_type)


def art_apgd_minimal_l1():
    name = 'apgd_minimal'
    source = 'art'
    threat_model = 'l1'
    step_size = 0.1
    num_steps = 100
    nb_random_init = 5
    loss_type = None

    init_eps = 10  # initial guess for line search
    search_steps = 20  # number of search steps for line + binary search


def art_apgd_minimal_l2():
    name = 'apgd_minimal'
    source = 'art'
    threat_model = 'l2'
    step_size = 0.1
    num_steps = 100
    nb_random_init = 5
    loss_type = None

    init_eps = 1  # initial guess for line search
    search_steps = 20  # number of search steps for line + binary search


def art_apgd_minimal_linf():
    name = 'apgd_minimal'
    source = 'art'
    threat_model = 'linf'
    step_size = 0.1
    num_steps = 100
    nb_random_init = 5
    loss_type = None

    init_eps = 1 / 255  # initial guess for line search
    search_steps = 20  # number of search steps for line + binary search


def get_art_apgd_minimal(threat_model: str, step_size: float, num_steps: int, nb_random_init: int,
                         loss_type: Optional[str], init_eps: float, search_steps: int) -> Callable:
    attack = partial(AutoProjectedGradientDescent, norm=_norms[threat_model], eps_step=step_size,
                     max_iter=num_steps, nb_random_init=nb_random_init, loss_type=loss_type)
    max_eps = 1 if threat_model == 'linf' else None
    return ArtMinimalWrapper(attack=attack, init_eps=init_eps, max_eps=max_eps, search_steps=search_steps)


def art_bim():
    name = 'bim'
    source = 'art'
    threat_model = 'linf'
    epsilon = 0.3
    step_size = 0.1
    num_steps = 100


def get_art_bim(epsilon: float, step_size: float, num_steps: int) -> Callable:
    return partial(BasicIterativeMethod, eps=epsilon, eps_step=step_size, max_iter=num_steps)


def art_bim_minimal():
    name = 'bim_minimal'
    source = 'art'
    threat_model = 'linf'
    epsilon = 0.3
    step_size = 0.1
    num_steps = 100

    init_eps = 1 / 255  # initial guess for line search
    search_steps = 20  # number of search steps for line + binary search


def get_art_bim_minimal(threat_model: str, epsilon: float, step_size: float, num_steps: int,
                        init_eps: float, search_steps: int) -> Callable:
    attack = partial(BasicIterativeMethod, eps=epsilon, eps_step=step_size, max_iter=num_steps)
    max_eps = 1 if threat_model == 'linf' else None
    return ArtMinimalWrapper(attack=attack, init_eps=init_eps, max_eps=max_eps, search_steps=search_steps)


def art_bb():
    name = 'bb'
    source = 'art'
    threat_model = 'linf'
    overshoot = 1.1
    num_steps = 1000
    step_size = 1e-3
    lr_decay = 0.5
    lr_num_decay = 20
    momentum = 0.8
    num_binary_search_steps = 10
    init_size = 100


def get_art_bb(threat_model: str, overshoot: float, num_steps: int, step_size: float, lr_decay: float,
               lr_num_decay: int, momentum: float, num_binary_search_steps: int, init_size: int) -> Callable:
    return partial(BrendelBethgeAttack, norm=_norms[threat_model], overshoot=overshoot, steps=num_steps, lr=step_size,
                   lr_decay=lr_decay, lr_num_decay=lr_num_decay, momentum=momentum,
                   binary_search_steps=num_binary_search_steps, init_size=init_size)


def art_cw_l2():
    name = 'cw_l2'
    source = 'art'
    threat_model = 'l2'
    confidence = 0.0
    step_size = 0.01
    num_binary_search_steps = 10
    num_steps = 1000  # default was 10
    initial_const = 0.01
    max_halving = 5
    max_doubling = 5


def get_art_cw_l2(confidence: float, step_size: float, num_binary_search_steps: int, num_steps: int,
                  initial_const: float, max_halving: int, max_doubling: int) -> Callable:
    return partial(CarliniL2Method, confidence=confidence, learning_rate=step_size,
                   binary_search_steps=num_binary_search_steps, max_iter=num_steps, initial_const=initial_const,
                   max_halving=max_halving, max_doubling=max_doubling)


def art_cw_linf():
    name = 'cw_linf'
    source = 'art'
    threat_model = 'linf'
    confidence = 0.0
    step_size = 0.01
    num_steps = 10
    decrease_factor = 0.9
    initial_const = 0.01
    largest_const = 20.0
    const_factor = 2.0


def get_art_cw_linf(confidence: float, step_size: float, num_steps: int, decrease_factor: float,
                    initial_const: float, largest_const: float, const_factor: float) -> Callable:
    return partial(CarliniLInfMethod, confidence=confidence, learning_rate=step_size, max_iter=num_steps,
                   decrease_factor=decrease_factor, initial_const=initial_const, largest_const=largest_const,
                   const_factor=const_factor)


def art_deepfool():
    name = 'deepfool'
    source = 'art'
    threat_model = 'l2'
    num_steps = 100
    epsilon = 1e-6
    nb_grads = 10


def get_art_deepfool(num_steps: int, epsilon: float, nb_grads: int) -> Callable:
    return partial(DeepFool, max_iter=num_steps, epsilon=epsilon, nb_grads=nb_grads)


def art_ead():
    name = 'ead'
    source = 'art'
    threat_model = 'l1'
    confidence = 0.0
    step_size = 1e-2
    num_binary_search_steps = 9
    num_steps = 100
    beta = 1e-3
    initial_const = 1e-3
    decision_rule = 'EN'


def get_art_ead(confidence: float, step_size: float, num_binary_search_steps: int, num_steps: int, beta: float,
                initial_const: float, decision_rule: str) -> Callable:
    return partial(ElasticNet, confidence=confidence, learning_rate=step_size,
                   binary_search_steps=num_binary_search_steps, max_iter=num_steps, beta=beta,
                   initial_const=initial_const, decision_rule=decision_rule)


def art_fgm():
    name = 'fgm'
    source = 'art'
    threat_model = 'linf'
    epsilon = 0.3
    step_size = 0.1
    num_random_init = 0
    minimal = False


def get_art_fgm(threat_model: str, epsilon: float, step_size: float, num_random_init: int, minimal: bool) -> Callable:
    return partial(FastGradientMethod, norm=_norms[threat_model], eps=epsilon, eps_step=step_size,
                   num_random_init=num_random_init, minimal=minimal)


def art_fgm_minimal_l1():
    name = 'fgm_minimal'
    source = 'art'
    threat_model = 'l1'
    epsilon = 0.3
    step_size = 0.1
    num_random_init = 0
    minimal = False

    init_eps = 10  # initial guess for line search
    search_steps = 20  # number of search steps for line + binary search


def art_fgm_minimal_l2():
    name = 'fgm_minimal'
    source = 'art'
    threat_model = 'l2'
    epsilon = 0.3
    step_size = 0.1
    num_random_init = 0
    minimal = False

    init_eps = 1  # initial guess for line search
    search_steps = 20  # number of search steps for line + binary search


def art_fgm_minimal_linf():
    name = 'fgm_minimal'
    source = 'art'
    threat_model = 'linf'
    epsilon = 0.3
    step_size = 0.1
    num_random_init = 0
    minimal = False

    init_eps = 1 / 255  # initial guess for line search
    search_steps = 20  # number of search steps for line + binary search


def get_art_fgm_minimal(threat_model: str, epsilon: float, step_size: float, num_random_init: int, minimal: bool,
                         init_eps: float, search_steps: int) -> Callable:
    attack = partial(FastGradientMethod, norm=_norms[threat_model], eps=epsilon, eps_step=step_size,
                     num_random_init=num_random_init, minimal=minimal)
    max_eps = 1 if threat_model == 'linf' else None
    return ArtMinimalWrapper(attack=attack, init_eps=init_eps, max_eps=max_eps, search_steps=search_steps)


def art_jsma():
    name = 'jsma'
    source = 'art'
    threat_model = 'l1'
    theta = 0.1
    gamma = 1.0


def get_art_jsma(theta: float, gamma: float) -> Callable:
    return partial(SaliencyMapMethod, theta=theta, gamma=gamma)


def art_pgd():
    name = 'pgd'
    source = 'art'
    threat_model = 'linf'
    epsilon = 0.3
    step_size = 0.1
    num_steps = 40  # default was 100. We decided to keep the original num_steps reported in the paper
    num_random_init = 0
    random_eps = False


def get_art_pgd(threat_model: str, epsilon: float, step_size: float, num_steps: int, num_random_init: int,
                random_eps: bool) -> Callable:
    return partial(ProjectedGradientDescent, norm=_norms[threat_model], eps=epsilon, eps_step=step_size,
                   num_random_init=num_random_init, max_iter=num_steps, random_eps=random_eps)


def art_pgd_minimal_l1():
    name = 'pgd_minimal'
    source = 'art'
    threat_model = 'l1'
    step_size = 0.1
    num_steps = 40  # default was 100. We decided to keep the original num_steps reported in the paper
    num_random_init = 0
    random_eps = False

    init_eps = 10  # initial guess for line search
    search_steps = 20  # number of search steps for line + binary search


def art_pgd_minimal_l2():
    name = 'pgd_minimal'
    source = 'art'
    threat_model = 'l2'
    step_size = 0.1
    num_steps = 40  # default was 100. We decided to keep the original num_steps reported in the paper
    num_random_init = 0
    random_eps = False

    init_eps = 1  # initial guess for line search
    search_steps = 20  # number of search steps for line + binary search


def art_pgd_minimal_linf():
    name = 'pgd_minimal'
    source = 'art'
    threat_model = 'linf'
    step_size = 0.1
    num_steps = 40  # default was 100. We decided to keep the original num_steps reported in the paper
    num_random_init = 0
    random_eps = False

    init_eps = 1 / 255  # initial guess for line search
    search_steps = 20  # number of search steps for line + binary search


def get_art_pgd_minimal(threat_model: str, step_size: float, num_steps: int, num_random_init: int,
                        random_eps: bool, init_eps: float, search_steps: int) -> Callable:
    attack = partial(ProjectedGradientDescent, norm=_norms[threat_model], eps_step=step_size,
                     num_random_init=num_random_init, max_iter=num_steps, random_eps=random_eps)
    max_eps = 1 if threat_model == 'linf' else None
    return ArtMinimalWrapper(attack=attack, init_eps=init_eps, max_eps=max_eps, search_steps=search_steps)
