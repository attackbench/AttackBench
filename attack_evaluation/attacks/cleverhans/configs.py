from functools import partial
from typing import Callable

from cleverhans.torch.attacks.carlini_wagner_l2 import carlini_wagner_l2
from cleverhans.torch.attacks.fast_gradient_method import fast_gradient_method
from cleverhans.torch.attacks.hop_skip_jump_attack import hop_skip_jump_attack
from cleverhans.torch.attacks.projected_gradient_descent import projected_gradient_descent
from cleverhans.torch.attacks.spsa import spsa

from .wrapper import cleverhans_minimal_wrapper, cleverhans_wrapper

_prefix = 'ch'
_wrapper = cleverhans_wrapper
_norms = {
    'l0': 0,
    'l1': 1,
    'l2': 2,
    'linf': float('inf'),
}


def ch_cw_l2():
    name = 'cw_l2'
    source = 'cleverhans'
    threat_model = 'l2'
    lr = 5e-03
    confidence = 0
    clip_min = 0
    clip_max = 1
    initial_const = 1e-02
    binary_search_steps = 5
    steps = 100  # 1000


def get_ch_cw(lr: float, confidence: float, clip_min: float, clip_max: float, initial_const: float,
              binary_search_steps: int, steps: int) -> Callable:
    return partial(carlini_wagner_l2, max_iterations=steps, lr=lr, confidence=confidence, clip_max=clip_max,
                   clip_min=clip_min, initial_const=initial_const, binary_search_steps=binary_search_steps)


def ch_fgm():
    name = 'fgm'
    source = 'cleverhans'
    threat_model = 'l2'  # available: l1, l2, linf
    eps = 0.3


def get_ch_fgm(threat_model: str, eps: float) -> Callable:
    return partial(fast_gradient_method, norm=_norms[threat_model], eps=eps, clip_min=0, clip_max=1)


def ch_fgm_minimal_l1():
    name = 'fgm_minimal'
    source = 'cleverhans'
    threat_model = 'l1'  # available: l1, l2, linf

    init_eps = 10  # initial guess for line search
    search_steps = 20  # number of search steps for line + binary search


def ch_fgm_minimal_l2():
    name = 'fgm_minimal'
    source = 'cleverhans'
    threat_model = 'l2'  # available: l1, l2, linf

    init_eps = 1  # initial guess for line search
    search_steps = 20  # number of search steps for line + binary search


def ch_fgm_minimal_linf():
    name = 'fgm_minimal'
    source = 'cleverhans'
    threat_model = 'linf'  # available: l1, l2, linf

    init_eps = 1 / 255  # initial guess for line search
    search_steps = 20  # number of search steps for line + binary search


def get_ch_fgm_minimal(threat_model: str, init_eps: float, search_steps: int) -> Callable:
    attack = partial(fast_gradient_method, norm=_norms[threat_model], clip_min=0, clip_max=1)
    max_eps = 1 if threat_model == 'linf' else None
    return partial(cleverhans_minimal_wrapper, attack=attack, init_eps=init_eps, max_eps=max_eps,
                   search_steps=search_steps)


def ch_hsja():
    name = 'hsja'
    source = 'cleverhans'
    threat_model = 'l2'  # available: l2, linf
    steps = 64
    initial_num_evals = 100
    max_num_evals = 10000
    stepsize_search = "geometric_progression"
    gamma = 1.0
    constraint = 2
    batch_size = 128


def get_ch_hsja(threat_model: str, steps: int, initial_num_evals: int, max_num_evals: int, stepsize_search: int,
                gamma: float, constraint: int, batch_size: int) -> Callable:
    return partial(hop_skip_jump_attack, norm=_norms[threat_model], initial_num_evals=initial_num_evals,
                   max_num_evals=max_num_evals, stepsize_search=stepsize_search, num_iterations=steps,
                   gamma=gamma, constraint=constraint, batch_size=batch_size, clip_min=0, clip_max=1)


def ch_spsa():
    name = 'spsa'
    source = 'cleverhans'
    steps = 100
    threat_model = 'linf'
    eps = 0.3
    early_stop_loss_threshold = None
    lr = 0.01
    delta = 0.01
    spsa_samples = 128
    spsa_iters = 1


def get_ch_spsa(threat_model: str, steps, eps: float, early_stop_loss_threshold: float, lr: float, delta: float,
                spsa_samples: int, spsa_iters: int) -> Callable:
    return partial(spsa, norm=_norms[threat_model], nb_iter=steps, eps=eps, learning_rate=lr, delta=delta,
                   spsa_iters=spsa_iters, early_stop_loss_threshold=early_stop_loss_threshold,
                   spsa_samples=spsa_samples, clip_min=0, clip_max=1, sanity_checks=False)


def ch_pgd():
    name = 'pgd'
    source = 'cleverhans'
    threat_model = 'l2'  # available: np.inf, 1 or 2.
    eps = 10.0
    eps_iter = 1.0
    steps = 20


def get_ch_pgd(threat_model: str, eps: float, eps_iter: float, steps: int) -> Callable:
    return partial(projected_gradient_descent, norm=_norms[threat_model], nb_iter=steps, eps=eps, eps_iter=eps_iter,
                   clip_min=0, clip_max=1, sanity_checks=False)


def ch_pgd_minimal_l1():
    name = 'pgd_minimal'
    source = 'cleverhans'
    threat_model = 'l1'  # available: np.inf, 1 or 2.
    eps_iter = 1.0
    steps = 40 # default was 20

    init_eps = 10  # initial guess for line search
    search_steps = 20  # number of search steps for line + binary search


def ch_pgd_minimal_l2():
    name = 'pgd_minimal'
    source = 'cleverhans'
    threat_model = 'l2'  # available: np.inf, 1 or 2.
    eps_iter = 1.0
    steps = 40 # default was 20

    init_eps = 1  # initial guess for line search
    search_steps = 20  # number of search steps for line + binary search


def ch_pgd_minimal_linf():
    name = 'pgd_minimal'
    source = 'cleverhans'
    threat_model = 'linf'  # available: np.inf, 1 or 2.
    eps_iter = 1.0
    steps = 40 # default was 20

    init_eps = 1 / 255  # initial guess for line search
    search_steps = 20  # number of search steps for line + binary search


def get_ch_pgd_minimal(threat_model: str, eps_iter: float, steps: int, init_eps: float, search_steps: int) -> Callable:
    attack = partial(projected_gradient_descent, norm=_norms[threat_model], nb_iter=steps, eps_iter=eps_iter,
                     clip_min=0, clip_max=1, sanity_checks=False)
    max_eps = 1 if threat_model == 'linf' else None
    return partial(cleverhans_minimal_wrapper, attack=attack, init_eps=init_eps, max_eps=max_eps,
                   search_steps=search_steps)
