from collections import OrderedDict
from pathlib import Path

import torch
from adv_lib.distances.lp_norms import l0_distances, l1_distances, l2_distances, linf_distances
from sacred import Experiment
from sacred.observers import FileStorageObserver

from attacks.ingredient import attack_ingredient, get_attack
from datasets.ingredient import dataset_ingredient, get_loader
from models.ingredient import get_model, model_ingredient
from utils import run_attack, set_seed

ex = Experiment('attack_evaluation', ingredients=[dataset_ingredient, model_ingredient, attack_ingredient])


@ex.config
def config():
    cpu = False  # force experiment to run on CPU
    save_adv = False  # save the inputs and perturbed inputs; not to be used with large datasets
    cudnn_flag = 'deterministic'  # choose between "deterministic" and "benchmark"


@ex.named_config
def save_adv():  # act as a flag
    save_adv = True


@ex.option_hook
def modify_filestorage(options):
    if (file_storage := options['--file_storage']) is None:
        return

    update = options['UPDATE']

    # find dataset, model and attack names from CLI
    names = []
    for ingredient in (dataset_ingredient, model_ingredient, attack_ingredient):
        ingredient_name = ingredient.path
        prefix = ingredient_name + '.'
        ingredient_updates = list(filter(lambda s: s.startswith(prefix) and '=' not in s, update))
        if (n := len(ingredient_updates)) != 1:
            raise ValueError(f'Incorrect {ingredient_name} configuration: {n} (!=1) named configs specified.')
        named_config = ingredient_updates[0].removeprefix(prefix)
        # names.append(ingredient.named_configs[named_config]()['name'])
        names.append(named_config)

    # find threat model
    attack_updates = list(filter(lambda s: s.startswith('attack.') and 'threat_model=' in s, update))
    if len(attack_updates):
        threat_model = attack_updates[-1].split('=')[-1]
    else:
        threat_model = ingredient.named_configs[named_config]()['threat_model']

    batch_size_update = list(filter(lambda s: "batch_size" in s, update))
    if len(batch_size_update):
        batch_size = batch_size_update[-1].split('=')[-1]
    else:
        batch_size = dataset_ingredient.named_configs[names[0]]()['batch_size']
    batch_name = f'batch_size_{batch_size}'

    # insert batch_size at desired position for folder structure
    names.insert(2, batch_name)

    # insert threat model at desired position for folder structure
    names.insert(1, threat_model)

    options['--file_storage'] = Path(file_storage).joinpath(*names).as_posix()


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
         _config, _run, _log, _seed):
    device = torch.device('cuda' if torch.cuda.is_available() and not cpu else 'cpu')
    setattr(torch.backends.cudnn, cudnn_flag, True)

    set_seed(_seed)

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
