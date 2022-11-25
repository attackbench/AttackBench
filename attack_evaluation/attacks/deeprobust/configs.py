import warnings
from functools import partial
from typing import Callable, Tuple

from deeprobust.image.attack.cw import CarliniWagner
from deeprobust.image.attack.deepfool import DeepFool
from deeprobust.image.attack.fgsm import FGSM
from deeprobust.image.attack.pgd import PGD

from .wrapper import DeepRobustMinimalWrapper
from ..utils import ConfigGetter


def dr_cw_l2():
    name = 'cw_l2'
    source = 'deeprobust'
    threat_model = 'l2'
    confidence = 0.0001
    num_binary_search_steps = 5
    num_steps = 100 # default was 1000
    initial_const = 0.01
    step_size = 0.00001
    abort_early = True


def get_dr_cw_l2(confidence: float, num_binary_search_steps: int, num_steps: int, initial_const: float,
                 step_size: float, abort_early: bool) -> Tuple[Callable, dict]:
    warnings.warn('C&W L2 is not functional with DeepRobust.')
    # Several issues:
    #  - cannot import _status_message from scipy.optimize.optimize with scipy>=1.8.0 ==> use scipy<1.8.0
    #  - does not work in batches
    #  - error in line 181 of deeprobust.image.attack.cw which tries to format the loss tensor
    return CarliniWagner, dict(confidence=confidence, binary_search_steps=num_binary_search_steps,
                               abort_early=abort_early, max_iterations=num_steps, initial_const=initial_const,
                               learning_rate=step_size)


def dr_deepfool():
    name = 'deepfool'
    source = 'deeprobust'
    threat_model = 'l2'
    overshoot = 0.02
    num_steps = 50
    num_classes = 10


def get_dr_deepfool(overshoot: float, num_steps: int, num_classes: int) -> Tuple[Callable, dict]:
    warnings.warn('DeepFool does not support batches in DeepRobust.')
    return DeepFool, dict(overshoot=overshoot, max_iteration=num_steps, num_classes=num_classes)


def dr_pgd():
    name = 'pgd'
    source = 'deeprobust'
    threat_model = 'linf'
    epsilon = 0.3
    num_steps = 40
    step_size = 0.01


def get_dr_pgd(threat_model: str, epsilon: float, num_steps: int, step_size: float) -> Tuple[Callable, dict]:
    return PGD, dict(bound=threat_model, epsilon=epsilon, num_steps=num_steps, step_size=step_size)


def dr_pgd_minimal():
    name = 'pgd_minimal'
    source = 'deeprobust'
    threat_model = 'linf'
    num_steps = 40
    step_size = 0.01

    init_eps = 1  # initial guess for line search
    search_steps = 20  # number of search steps for line + binary search


def get_dr_pgd_minimal(threat_model: str, num_steps: int, step_size: float, init_eps: float,
                       search_steps: int) -> Tuple[Callable, dict]:
    max_eps = 1 if threat_model == 'linf' else None
    attack = partial(DeepRobustMinimalWrapper, attack=PGD, init_eps=init_eps, search_steps=search_steps,
                     max_eps=max_eps)
    return attack, dict(bound=threat_model, num_steps=num_steps, step_size=step_size)


def dr_fgm():
    name = 'fgm'
    source = 'deeprobust'
    threat_model = 'linf'  # [linf, l2]
    epsilon = 0.2


def get_dr_fgm(threat_model: str, epsilon: float) -> Tuple[Callable, dict]:
    order = {'linf': float('inf'), 'l2': 2}
    return FGSM, dict(order=order[threat_model], epsilon=epsilon, clip_max=1, clip_min=0)


def dr_fgm_minimal():
    name = 'fgm_minimal'
    source = 'deeprobust'
    threat_model = 'linf'  # [linf, l2]

    init_eps = 1  # initial guess for line search
    search_steps = 20  # number of search steps for line + binary search


def get_dr_fgm_minimal(threat_model: str, init_eps: float, search_steps: int) -> Tuple[Callable, dict]:
    order = {'linf': float('inf'), 'l2': 2}
    max_eps = 1 if threat_model == 'linf' else None
    attack = partial(DeepRobustMinimalWrapper, attack=FGSM, init_eps=init_eps, search_steps=search_steps,
                     max_eps=max_eps)
    return attack, dict(order=order[threat_model], clip_min=0, clip_max=1)


deeprobust_index = {
    'cw_l2': ConfigGetter(config=dr_cw_l2, getter=get_dr_cw_l2),
    'deepfool': ConfigGetter(config=dr_deepfool, getter=get_dr_deepfool),
    'pgd': ConfigGetter(config=dr_pgd, getter=get_dr_pgd),
    'pgd_minimal': ConfigGetter(config=dr_pgd_minimal, getter=get_dr_pgd_minimal),
    'fgm': ConfigGetter(config=dr_fgm, getter=get_dr_fgm),
    'fgm_minimal': ConfigGetter(config=dr_fgm_minimal, getter=get_dr_fgm_minimal),
}
