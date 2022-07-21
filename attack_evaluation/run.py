from collections import OrderedDict
from pathlib import Path
from pprint import pformat

import torch
from adv_lib.distances.lp_norms import l0_distances, l1_distances, l2_distances, linf_distances
from sacred import Experiment
from sacred.observers import FileStorageObserver

from attacks.ingredient import attack_ingredient, get_attack
from datasets.ingredient import dataset_ingredient, get_loader
from models.ingredient import get_model, model_ingredient
from utils import run_attack

ex = Experiment('attack_evaluation', ingredients=[dataset_ingredient, model_ingredient, attack_ingredient])


@ex.config
def config():
    cpu = False  # force experiment to run on CPU
    save_adv = False  # save the inputs and perturbed inputs; not to be used with large datasets
    cudnn_flag = 'deterministic'  # choose between "deterministic" and "benchmark"


@ex.named_config
def save_adv():  # act as a flag
    save_adv = True


metrics = OrderedDict([
    ('linf', linf_distances),
    ('l0', l0_distances),
    ('l1', l1_distances),
    ('l2', l2_distances),
])


@ex.automain
def main(cpu: bool,
         cudnn_flag: str,
         save_adv: bool,
         _config, _run, _log):

    device = torch.device('cuda' if torch.cuda.is_available() and not cpu else 'cpu')
    setattr(torch.backends.cudnn, cudnn_flag, True)

    loader = get_loader()
    attack = get_attack()
    model = get_model()
    model.to(device)

    # find the current folder where the artifacts are saved
    file_observers = [obs for obs in _run.observers if isinstance(obs, FileStorageObserver)]
    save_dir = file_observers[0].dir if len(file_observers) else None

    attack_data = run_attack(model=model, loader=loader, attack=attack, metrics=metrics,
                             return_adv=save_adv and save_dir is not None)

    if save_adv and save_dir is not None:
        torch.save(attack_data, Path(save_dir) / f'attack_data.pt')

    if 'inputs' in attack_data.keys():
        del attack_data['inputs'], attack_data['adv_inputs']
    _run.info = attack_data
    # _log.info(pformat(attack_data, compact=True))
