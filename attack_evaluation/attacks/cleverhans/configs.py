from functools import partial
from typing import Callable, Optional

from cleverhans.torch.attacks.carlini_wagner_l2 import carlini_wagner_l2
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.hop_skip_jump_attack import hop_skip_jump_attack
from cleverhans.torch.attacks.spsa import spsa
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.torch.attacks.sparse_l1_descent import sparse_l1_descent
from ..utils import ConfigGetter


def ch_cw_l2():
    name = 'cw_l2'
    source = 'cleverhans'
    norm = 2
    lr = 5e-03
    confidence = 0
    clip_min = 0
    clip_max = 1
    initial_const = 1e-02
    binary_search_steps = 5
    steps = 1000


def get_ch_cw(norm: float, lr: float, confidence: float, clip_min: float, clip_max: float,
              initial_const: float, binary_search_steps: int, steps: int) -> Callable:
    return partial(carlini_wagner_l2, max_iterations=steps, lr=lr, confidence=confidence,
                   clip_max=clip_max, clip_min=clip_min, initial_const=initial_const,
                   binary_search_steps=binary_search_steps)


def ch_fgm():
    name = 'fgm'
    source = 'cleverhans'
    norm = 2  # available: l1, l2, linf
    eps = 0.3
    clip_min = 0
    clip_max = 1


def get_ch_fgm(norm: float, eps: float, clip_min: float, clip_max: float) -> Callable:
    return partial(fast_gradient_method, norm=norm, eps=eps, clip_min=clip_min, clip_max=clip_max)


def ch_hsja():
    name = 'hsja'
    source = 'cleverhans'
    norm = 2  # available: l2, linf
    steps = 64
    initial_num_evals = 100
    max_num_evals = 10000
    stepsize_search = "geometric_progression"
    gamma = 1.0
    constraint = 2
    batch_size = 128
    clip_min = 0
    clip_max = 1


def get_ch_hsja(norm: float, steps: int, initial_num_evals: int, max_num_evals: int, stepsize_search: int, gamma: float,
                constraint: int, batch_size: int, clip_min: float, clip_max: float) -> Callable:
    return partial(hop_skip_jump_attack, norm=norm, initial_num_evals=initial_num_evals, max_num_evals=max_num_evals,
                   stepsize_search=stepsize_search, num_iterations=steps, gamma=gamma, constraint=constraint,
                   batch_size=batch_size, clip_min=clip_min, clip_max=clip_max)


def ch_spsa():
    name = 'spsa'
    source = 'cleverhans'
    steps = 100
    norm = float('inf')
    eps = 0.3
    clip_min = -float('inf')
    clip_max = float('inf')
    early_stop_loss_threshold = None
    lr = 0.01
    delta = 0.01
    spsa_samples = 128
    spsa_iters = 1


def get_ch_spsa(norm: float, steps, eps: float, clip_min: float, clip_max: float, early_stop_loss_threshold: float,
                lr: float, delta: float, spsa_samples: int, spsa_iters: int) -> Callable:
    return partial(spsa, norm=norm, nb_iter=steps, eps=eps, learning_rate=lr, delta=delta, spsa_iters=spsa_iters,
                   early_stop_loss_threshold=early_stop_loss_threshold, spsa_samples=spsa_samples,
                   clip_min=clip_min, clip_max=clip_max, sanity_checks=False)


def ch_pgd():
    name = 'pgd'
    source = 'cleverhans'
    norm = 2  # available: np.inf, 1 or 2.
    eps = 10.0
    eps_iter = 1.0
    steps = 20
    clip_min = None
    clip_max = None


def get_ch_pgd(norm: float, eps: float, eps_iter: float, steps: int, clip_min: float, clip_max: float) -> Callable:
    return partial(projected_gradient_descent, norm=norm, nb_iter=steps, eps=eps, eps_iter=eps_iter,
                   clip_min=clip_min, clip_max=clip_max, sanity_checks=False)


cleverhans_index = {
    'cw_l2': ConfigGetter(config=ch_cw_l2, getter=get_ch_cw),
    'fgm': ConfigGetter(config=ch_fgm, getter=get_ch_fgm),
    'hsja': ConfigGetter(config=ch_hsja, getter=get_ch_hsja),
    'spsa': ConfigGetter(config=ch_spsa, getter=get_ch_spsa),
    'pgd': ConfigGetter(config=ch_pgd, getter=get_ch_pgd)
}