import json
import warnings
from pathlib import Path
from typing import Mapping, Tuple, Union

import numpy as np

from utils import Scenario


def read_results(info_file: Union[Path, str],
                 already_adv_distance: float = 0,
                 worst_case_distance: float = float('inf')) -> Tuple[Scenario, Mapping[str, float]]:
    info_file = Path(info_file)
    assert info_file.exists(), f'No info.json found in {dir}'

    # read config file
    with open(info_file.parent / 'config.json', 'r') as f:
        config = json.load(f)

    # extract main configs
    dataset = config['dataset']['name']
    model = config['model']['name']
    threat_model = config['attack']['threat_model']

    # read info file
    with open(info_file, 'r') as f:
        info = json.load(f)

    # get hashes and distances for the adversarial examples
    hashes = info['hashes']
    distances = np.array(info['distances'][threat_model])
    ori_success = np.array(info['ori_success'])
    adv_success = np.array(info['adv_success'])

    # check that adversarial examples have 0 distance for adversarial clean samples
    if (n := np.count_nonzero(distances[ori_success])):
        warnings.warn(f'{n} already adversarial clean samples have non zero perturbations in {info_file}.')

    # replace distances with inf for failed attacks and 0 for already adv clean samples
    distances[~adv_success] = worst_case_distance
    distances[ori_success] = already_adv_distance

    # store results
    scenario = Scenario(dataset=dataset, threat_model=threat_model, model=model)
    return scenario, {hash: distance for (hash, distance) in zip(hashes, distances)}