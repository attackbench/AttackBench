from collections import namedtuple

import numpy as np
import json
from itertools import product

Scenario = namedtuple('Scenario', ['dataset', 'batch_size', 'threat_model', 'model'])

_MAX_GAIN = 2.1


def top_k_attacks(attacks_to_plot: dict, k: int = None, only_distinct: bool = False):
    if only_distinct:
        return top_k_unique_attacks(attacks_to_plot, k)
    #attack_names = [k for k, v in sorted(attacks_to_plot.items(), reverse=False, key=lambda item: item[1]['area'])]
    attack_names = [k for k, v in sorted(attacks_to_plot.items(), reverse=True, key=lambda item: item[1]['optimality'])]
    return zip(attack_names, [rename_attack(atk) for atk in attack_names][:k])

def top_k_unique_attacks(attacks_to_plot: dict, k: int = None):
    # {k: v['area'] for k, v in sorted(attacks_to_plot.items(), reverse=True, key=lambda item: item[1]['area'])}
    attack_names = np.array([k for k, v in sorted(attacks_to_plot.items(), reverse=False, key=lambda item: item[1]['area'])])
    attack_no_lib = [rename_attack(atk).split(' ')[1] for atk in attack_names]
    names, index = np.unique(attack_no_lib, return_index=True)
    return zip(attack_names[index][:k], names[:k])


def rename_attack(atk_name):
    return ((atk_name[:-2].replace('_minimal', '').replace('apgd_t', 'APGD$_t$').replace('l1', '$\ell_1$'))
            .replace('linf', '$\ell_\infty$').replace('l2', '$\ell_\infty$').replace('_$\ell','-$\ell').replace('apgd', 'APGD').replace('fmn', 'FMN').replace('pdpgd', 'PDPGD')
            .replace('sigma_zero', '$\sigma$-zero').replace('bb', 'BB').replace('pgd0', 'PGD-$\ell_0$').replace('ead', 'EAD')
            .replace('vfga', 'VFGA').replace('ddn_NQ', 'DDN').replace('ddn', 'DDN').replace('bim', 'BIM').replace('alma', 'ALMA')
            .replace('adv_lib_', 'AdvLib ').replace('original_', 'Original ').replace('fb_', 'Foolbox ').replace('ta_', 'TorchAttack ')
            .replace('art_', 'Art ').replace('dr_', 'DeepRobust ').replace('ch_', 'Cleverhans ')
            .replace('fab', 'FAB').replace('pgd', 'PGD').replace('cw', 'CW').replace('pdgd', 'PDGD')
            )

def get_model_key(model):
    with open('exp_configs/cifar10_models_key.json', 'r') as f:
        models_dict = json.load(f)
    return models_dict[model]


def complementarity(atk1: np.ndarray, atk2: np.ndarray) -> float:
    diversity = (atk1 ^ atk2).sum()
    double_fault = ((atk1 + atk2) == 0).sum()
    available = diversity + double_fault
    if available == 0:
        return 0
    return diversity / available


def ensemble_gain_(atk1: np.ndarray, atk2: np.ndarray) -> float:
    assert len(atk1) == len(atk2), "Invalid shape error when evaluating attacks gain."
    c = complementarity(atk1, atk2)
    if c == 0:
        return 0

    e = 1 / (len(atk1) - max(atk1.sum(), atk2.sum()))
    if e == float('inf'):
        return _MAX_GAIN

    return min(c * e, _MAX_GAIN)


def ensemble_gain(atk1: np.ndarray, atk2: np.ndarray) -> float:
    n = len(atk1)
    return ((atk2 == 0) & (atk1 == 1)).sum() / n


def ensemble_distances(atk1_distances: np.ndarray, atk2_distances: np.ndarray) -> np.ndarray:
    return np.minimum(atk1_distances, atk2_distances)

def eval_optimality(adv_distances: np.ndarray, best_distances: list) -> np.ndarray:
    distances, counts = np.unique(best_distances, return_counts=True)
    robust_acc = 1 - counts.cumsum() / len(best_distances)

    # get quantities for optimality calculation
    clean_acc = np.count_nonzero(best_distances) / len(best_distances)
    max_dist = np.amax(distances)
    best_area = np.trapz(robust_acc, distances)

    distances, counts = np.unique(adv_distances, return_counts=True)
    robust_acc = 1 - counts.cumsum() / len(adv_distances)

    distances_clipped, counts = np.unique(adv_distances.clip(min=None, max=max_dist), return_counts=True)
    robust_acc_clipped = 1 - counts.cumsum() / len(adv_distances)

    area = np.trapz(robust_acc_clipped, distances_clipped)
    optimality = 1 - (area - best_area) / (clean_acc * max_dist - best_area)

    return optimality

