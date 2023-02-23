import warnings
from functools import partial
from typing import Dict, Optional

from deeprobust.image.attack.cw import CarliniWagner
from deeprobust.image.attack.deepfool import DeepFool
from deeprobust.image.attack.fgsm import FGSM
from deeprobust.image.attack.pgd import PGD

from .wrapper import DeepRobustMinimalWrapper, deeprobust_wrapper
from ..ingredient import minimal_init_eps, minimal_search_steps

_prefix = 'dr'
_wrapper = deeprobust_wrapper


def dr_cw_l2():
    name = 'cw_l2'
    source = 'deeprobust'
    threat_model = 'l2'
    confidence = 0.0001
    num_binary_search_steps = 5
    num_steps = 100  # default was 1000
    initial_const = 0.01
    step_size = 0.00001
    abort_early = True


def get_dr_cw_l2(confidence: float, num_binary_search_steps: int, num_steps: int, initial_const: float,
                 step_size: float, abort_early: bool) -> Dict:
    warnings.warn('C&W L2 is not functional with DeepRobust.')
    # Several issues:
    #  - cannot import _status_message from scipy.optimize.optimize with scipy>=1.8.0 ==> use scipy<1.8.0
    #  - does not work in batches
    #  - error in line 181 of deeprobust.image.attack.cw which tries to format the loss tensor
    return dict(
        attack=CarliniWagner,
        attack_params=dict(confidence=confidence, binary_search_steps=num_binary_search_steps, abort_early=abort_early,
                           max_iterations=num_steps, initial_const=initial_const, learning_rate=step_size)
    )


def dr_deepfool():
    name = 'deepfool'
    source = 'deeprobust'
    threat_model = 'l2'
    overshoot = 0.02
    num_steps = 50
    num_classes = 10


def get_dr_deepfool(overshoot: float, num_steps: int, num_classes: int) -> Dict:
    warnings.warn('DeepFool does not support batches in DeepRobust.')
    return dict(attack=DeepFool,
                attack_params=dict(overshoot=overshoot, max_iteration=num_steps, num_classes=num_classes))


def dr_pgd():
    name = 'pgd'
    source = 'deeprobust'
    threat_model = 'linf'  # available: 'l2', 'linf'
    epsilon = 0.3
    num_steps = 40
    step_size = 0.01


def get_dr_pgd(threat_model: str, epsilon: float, num_steps: int, step_size: float) -> Dict:
    return dict(attack=PGD,
                attack_params=dict(bound=threat_model, epsilon=epsilon, num_steps=num_steps, step_size=step_size))


def dr_pgd_minimal():
    name = 'pgd_minimal'
    source = 'deeprobust'
    threat_model = 'linf'  # available: 'l2', 'linf'
    num_steps = 40
    step_size = 0.01


def get_dr_pgd_minimal(threat_model: str, num_steps: int, step_size: float,
                       init_eps: Optional[float] = None, search_steps: int = minimal_search_steps) -> Dict:
    init_eps = minimal_init_eps[threat_model] if init_eps is None else init_eps
    max_eps = 1 if threat_model == 'linf' else None
    attack = partial(DeepRobustMinimalWrapper, attack=PGD, init_eps=init_eps, search_steps=search_steps,
                     max_eps=max_eps)
    return dict(attack=attack, attack_params=dict(bound=threat_model, num_steps=num_steps, step_size=step_size))


def dr_fgm():
    name = 'fgm'
    source = 'deeprobust'
    threat_model = 'linf'  # available: 'l2', 'linf'
    epsilon = 0.2


def get_dr_fgm(threat_model: str, epsilon: float) -> Dict:
    order = {'linf': float('inf'), 'l2': 2}
    return dict(attack=FGSM, attack_params=dict(order=order[threat_model], epsilon=epsilon, clip_max=1, clip_min=0))


def dr_fgm_minimal():
    name = 'fgm_minimal'
    source = 'deeprobust'
    threat_model = 'linf'  # available: 'l2', 'linf'


def get_dr_fgm_minimal(threat_model: str,
                       init_eps: Optional[float] = None, search_steps: int = minimal_search_steps) -> Dict:
    order = {'linf': float('inf'), 'l2': 2}
    init_eps = minimal_init_eps[threat_model] if init_eps is None else init_eps
    max_eps = 1 if threat_model == 'linf' else None
    attack = partial(DeepRobustMinimalWrapper, attack=FGSM, init_eps=init_eps, search_steps=search_steps,
                     max_eps=max_eps)
    return dict(attack=attack, attack_params=dict(order=order[threat_model], clip_min=0, clip_max=1))
