import inspect
import sys
from functools import partial
from typing import Callable, Optional

from foolbox.attacks import (
    L0BrendelBethgeAttack,
    L1BrendelBethgeAttack,
    L2BrendelBethgeAttack,
    LinfinityBrendelBethgeAttack,
    L2CarliniWagnerAttack,
    DatasetAttack,
    DDNAttack,
    L2DeepFoolAttack,
    LinfDeepFoolAttack,
    EADAttack,
    L0FMNAttack,
    L1FMNAttack,
    L2FMNAttack,
    LInfFMNAttack,
    L2ProjectedGradientDescentAttack,
    LinfProjectedGradientDescentAttack,
    L2FastGradientAttack,
    LinfFastGradientAttack,
    L2BasicIterativeAttack,
    LinfBasicIterativeAttack,
)

from .wrapper import FoolboxMinimalWrapper


def fb_bb():
    name = 'bb'
    source = 'foolbox'
    threat_model = 'linf'
    num_steps = 1000
    step_size = 0.001
    lr_decay = 0.5
    lr_num_decay = 20
    momentum = 0.8
    num_binary_search_steps = 10


_bb_attacks = {
    'l0': L0BrendelBethgeAttack,
    'l1': L1BrendelBethgeAttack,
    'l2': L2BrendelBethgeAttack,
    'linf': LinfinityBrendelBethgeAttack,
}


def get_fb_bb(threat_model: str, num_steps: int, step_size: float, lr_decay: float, lr_num_decay: float,
              momentum: float, num_binary_search_steps: int) -> Callable:
    return partial(_bb_attacks[threat_model], steps=num_steps, lr=step_size, lr_decay=lr_decay,
                   lr_num_decay=lr_num_decay, momentum=momentum, binary_search_steps=num_binary_search_steps)


def fb_cw_l2():
    name = 'cw_l2'
    source = 'foolbox'
    threat_model = 'l2'
    num_steps = 100  # default was 10000
    num_binary_search_steps = 9
    step_size = 0.01
    confidence = 0
    initial_const = 0.001
    abort_early = True


def get_fb_cw_l2(num_binary_search_steps: int, num_steps: int, step_size: float, confidence: float,
                 initial_const: float, abort_early: bool) -> Callable:
    return partial(L2CarliniWagnerAttack, binary_search_steps=num_binary_search_steps, steps=num_steps,
                   stepsize=step_size, confidence=confidence, initial_const=initial_const, abort_early=abort_early)


def fb_dataset():
    name = 'dataset'
    source = 'foolbox'
    threat_model = 'l2'


def get_fb_dataset():
    return partial(DatasetAttack)


def fb_ddn():
    name = 'ddn'
    source = 'foolbox'
    threat_model = 'l2'
    init_epsilon = 1
    num_steps = 100
    gamma = 0.05


def get_fb_ddn(init_epsilon: float, num_steps: int, gamma: float) -> Callable:
    return partial(DDNAttack, init_epsilon=init_epsilon, steps=num_steps, gamma=gamma)


def fb_deepfool():
    name = 'deepfool'
    source = 'foolbox'
    threat_model = 'linf'
    num_steps = 50
    candidates = 10
    overshoot = 0.02
    loss = 'logits'  # ∈ {'logits', 'crossentropy'}


_deepfool_attacks = {
    'l2': L2DeepFoolAttack,
    'linf': LinfDeepFoolAttack,
}


def get_fb_deepfool(threat_model: str, num_steps: int, candidates: int, overshoot: float, loss: str) -> Callable:
    return partial(_deepfool_attacks[threat_model], steps=num_steps, candidates=candidates, overshoot=overshoot,
                   loss=loss)


def fb_ead():
    name = 'ead'
    source = 'foolbox'
    threat_model = 'l1'
    num_binary_search_steps = 9
    num_steps = 10000
    step_size = 0.01
    confidence = 0
    initial_const = 0.001
    regularization = 0.01
    decision_rule = 'EN'  # ∈ {'EN', 'L1'}
    abort_early = True


def get_fb_ead(num_binary_search_steps: float, num_steps: int, step_size: float, confidence: float,
               initial_const: float, regularization: float, decision_rule: str, abort_early: bool) -> Callable:
    return partial(EADAttack, binary_search_steps=num_binary_search_steps, steps=num_steps, abort_early=abort_early,
                   initial_stepsize=step_size, confidence=confidence, initial_const=initial_const,
                   regularization=regularization, decision_rule=decision_rule)


def fb_fmn():
    name = 'fmn'
    source = 'foolbox'
    threat_model = 'linf'
    num_steps = 100
    max_stepsize = 1
    min_stepsize = None
    gamma = 0.05
    init_attack = None
    num_binary_search_steps = 10


_fmn_attacks = {
    'l0': L0FMNAttack,
    'l1': L1FMNAttack,
    'l2': L2FMNAttack,
    'linf': LInfFMNAttack
}


def get_fb_fmn(threat_model: str, num_steps: int, max_stepsize: float, gamma: float, min_stepsize: Optional[float],
               init_attack: Optional, num_binary_search_steps: int) -> Callable:
    return partial(_fmn_attacks[threat_model], steps=num_steps, max_stepsize=max_stepsize, min_stepsize=min_stepsize,
                   gamma=gamma, init_attack=init_attack, binary_search_steps=num_binary_search_steps)


def fb_pgd():
    name = 'pgd'
    source = 'foolbox'
    threat_model = 'l2'
    epsilon = 0.3
    num_steps = 50
    step_size = 0.025
    abs_stepsize = None


_pgd_attacks = {
    'l2': L2ProjectedGradientDescentAttack,
    'linf': LinfProjectedGradientDescentAttack,
}


def get_fb_pgd(threat_model: str, epsilon: float, num_steps: int, step_size: float, abs_stepsize: float) -> Callable:
    return partial(_pgd_attacks[threat_model], epsilon=epsilon, steps=num_steps, rel_stepsize=step_size,
                   abs_stepsize=abs_stepsize)


def fb_pgd_minimal():
    name = 'pgd_minimal'
    source = 'foolbox'
    threat_model = 'l2'
    num_steps = 50
    step_size = 0.025
    abs_stepsize = None

    init_eps = 1  # initial guess for line search
    search_steps = 20  # number of search steps for line + binary search


def get_fb_pgd_minimal(threat_model: str, num_steps: int, step_size: float, abs_stepsize: float,
                       init_eps: float, search_steps: int) -> Callable:
    max_eps = 1 if threat_model == 'linf' else None
    attack = partial(_pgd_attacks[threat_model], steps=num_steps, rel_stepsize=step_size, abs_stepsize=abs_stepsize)
    return partial(FoolboxMinimalWrapper, attack=attack, init_eps=init_eps, search_steps=search_steps, max_eps=max_eps)


def fb_fgm():
    name = 'fgm'
    source = 'foolbox'
    threat_model = 'l2'
    epsilon = 0.3


_fgm_attacks = {
    'l2': L2FastGradientAttack,
    'linf': LinfFastGradientAttack,
}


def get_fb_fgm(threat_model: str, epsilon: float) -> Callable:
    return partial(_fgm_attacks[threat_model], epsilon=epsilon)


def fb_fgm_minimal():
    name = 'fgm_minimal'
    source = 'foolbox'
    threat_model = 'l2'

    init_eps = 1  # initial guess for line search
    search_steps = 20  # number of search steps for line + binary search


def get_fb_fgm_minimal(threat_model: str, init_eps: float, search_steps: int) -> Callable:
    max_eps = 1 if threat_model == 'linf' else None
    attack = _fgm_attacks[threat_model]
    return partial(FoolboxMinimalWrapper, attack=attack, init_eps=init_eps, search_steps=search_steps, max_eps=max_eps)


def fb_bim():
    name = 'bim'
    source = 'foolbox'
    threat_model = 'linf'
    epsilon = 0.3
    num_steps = 10
    step_size = 0.2
    abs_stepsize = None


_bim_attacks = {
    'l2': L2BasicIterativeAttack,
    'linf': LinfBasicIterativeAttack,
}


def get_fb_bim(threat_model: str, epsilon: float, num_steps: int, step_size: float, abs_stepsize: float) -> Callable:
    return partial(_bim_attacks[threat_model], epsilon=epsilon, steps=num_steps, rel_stepsize=step_size,
                   abs_stepsize=abs_stepsize)


def fb_bim_minimal():
    name = 'bim_minimal'
    source = 'foolbox'
    threat_model = 'linf'
    num_steps = 10
    step_size = 0.2
    abs_stepsize = None

    init_eps = 1 / 255  # initial guess for line search
    search_steps = 20  # number of search steps for line + binary search


def get_fb_bim_minimal(threat_model: str, num_steps: int, step_size: float, abs_stepsize: float,
                       init_eps: float, search_steps: int) -> Callable:
    max_eps = 1 if threat_model == 'linf' else None
    attack = partial(_bim_attacks[threat_model], steps=num_steps, rel_stepsize=step_size, abs_stepsize=abs_stepsize)
    return partial(FoolboxMinimalWrapper, attack=attack, init_eps=init_eps, search_steps=search_steps, max_eps=max_eps)


foolbox_funcs = inspect.getmembers(sys.modules[__name__],
                                   predicate=lambda f: inspect.isfunction(f) and f.__module__ == __name__)
foolbox_configs, foolbox_getters = [], {}
for name, func in foolbox_funcs:
    if name.startswith('fb_'):
        foolbox_configs.append(func)
    elif name.startswith('get_fb_'):
        foolbox_getters[name.removeprefix('get_fb_')] = func
