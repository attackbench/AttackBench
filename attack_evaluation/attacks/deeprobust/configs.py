import warnings
from typing import Callable, Tuple

from deeprobust.image.attack.cw import CarliniWagner
from deeprobust.image.attack.pgd import PGD

from ..utils import ConfigGetter


def dr_cw_l2():
    name = 'cw_l2'
    source = 'deeprobust'
    confidence = 0.0001
    binary_search_steps = 5
    max_iterations = 1000
    initial_const = 0.01
    learning_rate = 0.00001
    abort_early = True


def get_dr_cw_l2(confidence: float, binary_search_steps: int, max_iterations: int, initial_const: float,
                 learning_rate: float, abort_early: bool) -> Tuple[Callable, dict]:
    warnings.warn('C&W L2 is not functional with DeepRobust.')
    # Several issues:
    #  - cannot import _status_message from scipy.optimize.optimize with scipy>=1.8.0 ==> use scipy<1.8.0
    #  - does not work in batches
    #  - error in line 181 of deeprobust.image.attack.cw which tries to format the loss tensor
    return CarliniWagner, dict(confidence=confidence, binary_search_steps=binary_search_steps, abort_early=abort_early,
                               max_iterations=max_iterations, initial_const=initial_const, learning_rate=learning_rate)


def dr_pgd():
    name = 'pgd'
    source = 'deeprobust'
    epsilon = 0.3
    num_steps = 40
    step_size = 0.01
    norm = float('inf')


def get_dr_pgd(norm: float, epsilon: float, num_steps: int, step_size: float) -> Tuple[Callable, dict]:
    bound = f'l{norm}'
    return PGD, dict(bound=bound, epsilon=epsilon, num_steps=num_steps, step_size=step_size)


deeprobust_index = {
    'cw_l2': ConfigGetter(config=dr_cw_l2, getter=get_dr_cw_l2),
    'pgd': ConfigGetter(config=dr_pgd, getter=get_dr_pgd),
}
