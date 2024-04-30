from collections import OrderedDict
from pathlib import Path
from pprint import pprint

import torch
from adv_lib.distances.lp_norms import l0_distances, l1_distances, l2_distances, linf_distances
from sacred import Experiment
from sacred.observers import FileStorageObserver

from .attacks.ingredient import attack_ingredient, get_attack
from .datasets.ingredient import dataset_ingredient, get_loader
from .models.ingredient import get_model, model_ingredient
from .utils import run_attack, set_seed

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
    names = {}
    for ingredient in (model_ingredient, attack_ingredient):
        ingredient_name = ingredient.path
        prefix = ingredient_name + '.'
        ingredient_updates = list(filter(lambda s: s.startswith(prefix) and '=' not in s, update))
        if (n := len(ingredient_updates)) != 1:
            raise ValueError(f'Incorrect {ingredient_name} configuration: {n} (!=1) named configs specified.')
        named_config = ingredient_updates[0].removeprefix(prefix)
        # names.append(ingredient.named_configs[named_config]()['name'])
        names[ingredient_name] = named_config

    # get dataset from model named config
    dataset = model_ingredient.named_configs[names['model']]()['dataset']

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
        batch_size = dataset_ingredient.configurations[0]()['batch_size']
    batch_name = f'batch_size_{batch_size}'

    # insert threat model and batch_size at desired position for folder structure
    subdirs = [dataset, threat_model, names['model'], batch_name, names['attack']]
    options['--file_storage'] = Path(file_storage).joinpath(*subdirs).as_posix()


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
    print(f'Running experiments with seed {_seed}')

    threat_model = _config['attack']['threat_model']
    loader = get_loader(dataset=_config['model']['dataset'])
    attack = get_attack()
    model = get_model()
    model.to(device)

    if len(loader) == 0:  # end experiment if there are no inputs to attack
        return

    # find the current folder where the artifacts are saved
    file_observers = [obs for obs in _run.observers if isinstance(obs, FileStorageObserver)]
    save_dir = file_observers[0].dir if len(file_observers) else None

    attack_data = run_attack(model=model, loader=loader, attack=attack, metrics=metrics, threat_model=threat_model,
                             return_adv=save_adv and save_dir is not None, debug=_run.debug)

    if save_adv and save_dir is not None:
        torch.save(attack_data, Path(save_dir) / f'attack_data.pt')

    if 'inputs' in attack_data.keys():
        del attack_data['inputs'], attack_data['adv_inputs']
    _run.info = attack_data

    if _run.debug:
        pprint(_run.info)

