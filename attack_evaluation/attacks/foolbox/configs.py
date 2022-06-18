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

from ..utils import ConfigGetter


def fb_bb():
    name = 'bb'
    source = 'foolbox'
    norm = float('inf')
    steps = 1000
    lr = 0.001
    lr_decay = 0.5
    lr_num_decay = 20
    momentum = 0.8
    binary_search_steps = 10


_bb_attacks = {
    0: L0BrendelBethgeAttack,
    1: L1BrendelBethgeAttack,
    2: L2BrendelBethgeAttack,
    float('inf'): LinfinityBrendelBethgeAttack,
}


def get_fb_bb(norm: float, steps: int, lr: float, lr_decay: float, lr_num_decay: float,
              momentum: float,
              binary_search_steps: int) -> Callable:
    return partial(_bb_attacks[float(norm)], steps=steps, lr=lr, lr_decay=lr_decay,
                   lr_num_decay=lr_num_decay,
                   momentum=momentum, binary_search_steps=binary_search_steps)


def fb_cw_l2():
    name = 'cw_l2'
    source = 'foolbox'
    binary_search_steps = 9
    steps = 10000
    stepsize = 0.01
    confidence = 0
    initial_const = 0.001
    abort_early = True


def get_fb_cw_l2(binary_search_steps: int, steps: int, stepsize: float, confidence: float,
                 initial_const: float,
                 abort_early: bool) -> Callable:
    return partial(L2CarliniWagnerAttack, binary_search_steps=binary_search_steps, steps=steps,
                   stepsize=stepsize,
                   confidence=confidence, initial_const=initial_const, abort_early=abort_early)


def fb_dataset():
    name = 'dataset'
    source = 'foolbox'


def get_fb_dataset():
    return partial(DatasetAttack)


def fb_ddn():
    name = 'ddn'
    source = 'foolbox'
    init_epsilon = 1
    steps = 100
    gamma = 0.05


def get_fb_ddn(init_epsilon: float, steps: int, gamma: float) -> Callable:
    return partial(DDNAttack, init_epsilon=init_epsilon, steps=steps, gamma=gamma)


def fb_deepfool():
    name = 'deepfool'
    source = 'foolbox'
    norm = float('inf')
    steps = 50
    candidates = 10
    overshoot = 0.02
    loss = 'logits'  # ∈ {'logits', 'crossentropy'}


_deepfool_attacks = {
    2: L2DeepFoolAttack,
    float('inf'): LinfDeepFoolAttack,
}


def get_fb_deepfool(norm: float, steps: int, candidates: int, overshoot: float,
                    loss: str) -> Callable:
    return partial(_deepfool_attacks[float(norm)], steps=steps, candidates=candidates,
                   overshoot=overshoot, loss=loss)


def fb_ead():
    name = 'ead'
    source = 'foolbox'
    binary_search_steps = 9
    steps = 10000
    initial_stepsize = 0.01
    confidence = 0
    initial_const = 0.001
    regularization = 0.01
    decision_rule = 'EN'  # ∈ {'EN', 'L1'}
    abort_early = True


def get_fb_ead(binary_search_steps: float, steps: int, initial_stepsize: float, confidence: float,
               initial_const: float,
               regularization: float, decision_rule: str, abort_early: bool) -> Callable:
    return partial(EADAttack, binary_search_steps=binary_search_steps, steps=steps,
                   initial_stepsize=initial_stepsize,
                   confidence=confidence, initial_const=initial_const,
                   regularization=regularization,
                   decision_rule=decision_rule, abort_early=abort_early)


def fb_fmn():
    name = 'fmn'
    source = 'foolbox'
    norm = float('inf')
    steps = 100
    max_stepsize = 1
    min_stepsize = None
    gamma = 0.05
    init_attack = None
    binary_search_steps = 10


_fmn_attacks = {
    0: L0FMNAttack,
    1: L1FMNAttack,
    2: L2FMNAttack,
    float('inf'): LInfFMNAttack
}


def get_fb_fmn(norm: float, steps: int, max_stepsize: float, gamma: float,
               min_stepsize: Optional[float],
               init_attack: Optional, binary_search_steps: int) -> Callable:
    return partial(_fmn_attacks[float(norm)], steps=steps, max_stepsize=max_stepsize,
                   min_stepsize=min_stepsize,
                   gamma=gamma, init_attack=init_attack, binary_search_steps=binary_search_steps)


def fb_pgd():
    name = 'pgd'
    source = 'foolbox'
    norm = 2
    steps = 50
    rel_stepsize = 0.025
    abs_stepsize = None


_pgd_attacks = {
    2: L2ProjectedGradientDescentAttack,
    float('inf'): LinfProjectedGradientDescentAttack,
}


def get_fb_pgd(norm: float, steps: int, rel_stepsize: float, abs_stepsize: float) -> Callable:
    return partial(_pgd_attacks[float(norm)], steps=steps, rel_stepsize=rel_stepsize,
                   abs_stepsize=abs_stepsize)


def fb_fgm():
    name = 'fgm'
    source = 'foolbox'


_fgm_attacks = {
    2: L2FastGradientAttack,
    float('inf'): LinfFastGradientAttack,
}


def get_fb_fgm(norm: float) -> Callable:
    return partial(_fgm_attacks[float(norm)])


def fb_bim():
    name = 'bim'
    source = 'foolbox'
    steps = 10
    rel_stepsize = 0.2
    abs_stepsize = None


_bim_attacks = {
    2: L2BasicIterativeAttack,
    float('inf'): LinfBasicIterativeAttack,
}


def get_fb_bim(norm: float, steps: int, rel_stepsize: float, abs_stepsize: float) -> Callable:
    return partial(_bim_attacks[float(norm)], steps=steps, rel_stepsize=rel_stepsize,
                   abs_stepsize=abs_stepsize)


foolbox_index = {
    'bb': ConfigGetter(config=fb_bb, getter=get_fb_bb),
    'cw_l2': ConfigGetter(config=fb_cw_l2, getter=get_fb_cw_l2),
    'dataset': ConfigGetter(config=fb_dataset, getter=get_fb_dataset),
    'ddn': ConfigGetter(config=fb_ddn, getter=get_fb_ddn),
    'deepfool': ConfigGetter(config=fb_deepfool, getter=get_fb_deepfool),
    'ead': ConfigGetter(config=fb_ead, getter=get_fb_ead),
    'fmn': ConfigGetter(config=fb_fmn, getter=get_fb_fmn),
    'pgd': ConfigGetter(config=fb_pgd, getter=get_fb_pgd),
    'fgm': ConfigGetter(config=fb_fgm, getter=get_fb_fgm),
    'bim': ConfigGetter(config=fb_bim, getter=get_fb_bim),
}
