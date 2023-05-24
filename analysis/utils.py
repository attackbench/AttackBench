from collections import namedtuple

import numpy as np
import json

Scenario = namedtuple('Scenario', ['dataset', 'batch_size', 'threat_model', 'model'])

_MAX_GAIN = 2.1


def top_k_attacks(attacks_to_plot: dict, k: int = None):
    # {k: v['area'] for k, v in sorted(attacks_to_plot.items(), reverse=True, key=lambda item: item[1]['area'])}
    return [k for k, v in sorted(attacks_to_plot.items(), reverse=False, key=lambda item: item[1]['area'])][:k]


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
