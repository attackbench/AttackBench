import argparse
import json
import math
import os
from collections import defaultdict
from copy import deepcopy
from importlib import resources
from itertools import product
from pathlib import Path

import numpy as np

import attack_evaluation
from attack_evaluation.run import ex

dataset_lengths = {
    'mnist': 10000,
    'cifar10': 10000,
    'cifar100': 10000,
    'imagenet': 50000,
}

if __name__ == '__main__':
    all_named_configs = list(ex.gather_named_configs())
    available_models = [name.removeprefix('model.') for name, config in all_named_configs if name.startswith('model.')]

    with resources.open_text(attack_evaluation, 'attacks.json') as f:
        attack_configs = json.load(f)

    available_libraries, available_threat_models, available_attacks = [], [], []
    for threat_model, libraries in attack_configs.items():
        available_threat_models.append(threat_model)
        for library, attacks in libraries.items():
            available_libraries.append(library)
            for attack in attacks['attacks']:
                available_attacks.append(attack)

    available_libraries = sorted(list(set(available_libraries)))
    available_attacks = sorted(list(set(available_attacks)))

    parser = argparse.ArgumentParser(description='Compute Canada Slurm runner for attack benchmark')

    # location args
    parser.add_argument('--result-dir', '-r', type=str, required=True, help='Directory where the results are saved')
    parser.add_argument('--job-dir', '-j', type=str, default=None,
                        help='Directory to store slurm scripts before submitting. Default to result-dir')

    # slurm args
    parser.add_argument('--account', type=str, default=None, help='Account allocation to use')
    parser.add_argument('--gpu-type', type=str, default=None, help='Device over which exp are executed')
    parser.add_argument('--gpu-count', type=int, default=1, help='Number of gpus for trial')
    parser.add_argument('--cpu-count', type=int, default=2, help='Number of cpus for trial')
    parser.add_argument('--memory', '--mem', type=int, default=16, help='Number of GB to allocate')
    parser.add_argument('--time', type=str, default='benchmark',
                        help='Job duration in DD-HH:MM format. "benchmark" to evaluate on 2 batches and extrapolate.')
    parser.add_argument('--min-time', type=int, default=120, help='Minimum time in seconds.')
    parser.add_argument('--environment', '--env', type=str, default=None, help='VirtualEnv to use')
    parser.add_argument('--submit', action='store_true', help='Submit the job to slurm')

    # benchmark args
    parser.add_argument('--num-batches', '--nb', type=int, default=3,
                        help='Number of batches to evaluate for time benchmark')
    parser.add_argument('--reduce-batches', type=int, default=4,
                        help='Use a smaller batch-size to quickly evaluate run-time')

    # benchmark args
    parser.add_argument('--config', type=str, required=True, help='Config file for the experiments to run')

    args = parser.parse_args()

    # exp setup
    exp_dir = Path(os.path.dirname(os.path.realpath(__file__)))
    result_dir = Path(args.result_dir)
    slurm_script_dir = Path(args.job_dir) if args.job_dir is not None else result_dir

    # read config
    config_file = Path(args.config)
    with open(config_file, 'r') as f:
        config = json.load(f)
    common_config_updates = config.pop('common')
    common_named_configs = common_config_updates.pop('named_configs', [])

    cartesian_config = config.pop('cartesian')
    ingredients = list(cartesian_config.keys())
    for ingredient, ingredient_config in cartesian_config.items():  # replace sub-config files
        if isinstance(ingredient_config, str) and ingredient_config.lower().endswith('.json'):
            if (subconfig_file := (config_file.parent / ingredient_config)).exists():
                with open(subconfig_file, 'r') as f:
                    cartesian_config[ingredient] = json.load(f)
            else:  # catch erroneous config file name
                raise ValueError(f'Configuration file {subconfig_file} does not exist.')
    ingredient_named_configs = list(cartesian_config.values())

    for i, ingredient_combination in enumerate(product(*ingredient_named_configs)):

        # generate combination specific named configs and config updates
        named_configs = []
        config_updates = defaultdict(dict)
        for ingredient, name in zip(ingredients, ingredient_combination):
            named_configs.append(f'{ingredient}.{name}')
            if isinstance(cartesian_config[ingredient], dict):
                config_updates[ingredient] = cartesian_config[ingredient][name]

        # merge with common named configs and config updates
        named_configs.extend(common_named_configs)
        for key, value in common_config_updates.items():
            if isinstance(value, dict):
                config_updates[key] = value | config_updates[key]
            else:
                config_updates[key] = value

        lines = [
            '#!/bin/bash',
            f'#SBATCH --output=%j_%x.out',
            f'#SBATCH --mem={args.memory}G',
            f'#SBATCH --cpus-per-task={args.cpu_count}',
        ]

        gpu_options = ['gpu', args.gpu_count]
        if args.gpu_type is not None:
            gpu_options.insert(1, args.gpu_type)
        lines.append(f'#SBATCH --gres={":".join(map(str, gpu_options))}')

        if args.account is not None:
            lines.append(f'#SBATCH --account={args.account}')

        if args.time == 'benchmark':
            run = ex.run(config_updates=config_updates, named_configs=named_configs,
                         options={'--loglevel': 'ERROR', '--queue': True})
            dataset = run.config['model']['dataset']
            run_num_samples = run.config['dataset']['num_samples']

            bench_batch_size = max(1, run.config['dataset']['batch_size'] // args.reduce_batches)
            bench_config_updates = deepcopy(config_updates)
            bench_config_updates['dataset']['batch_size'] = bench_batch_size
            bench_config_updates['dataset']['num_samples'] = args.num_batches * bench_batch_size
            try:
                run = ex.run(config_updates=bench_config_updates, named_configs=named_configs,
                             options={'--loglevel': 'ERROR'})
            except:
                print(f'Skipping {named_configs} | {config_updates} (crashed during benchmark).')
                continue

            times = np.sort(np.asarray(run.info['times']))
            time_per_batch = np.mean(times[1:-1])
            num_samples = run_num_samples if run_num_samples is not None else dataset_lengths[dataset]
            num_batches = math.ceil(min(num_samples, dataset_lengths[dataset]) / bench_batch_size)
            total_time = max(args.min_time, math.ceil(time_per_batch * num_batches * 1.1))  # add 10%
            hours, minutes, seconds = total_time // 3600, (total_time % 3600) // 60, total_time % 60
            time_string = f'{hours:02d}:{minutes:02d}:{seconds:02d}'
            print(f'Running {named_configs} | {config_updates} for {total_time}s = {time_string}')
            lines.append(f'#SBATCH --time={time_string}')
        else:
            lines.append(f'#SBATCH --time={args.time}')

        lines.extend(['module load python/3.9', f'cd {exp_dir.as_posix()}'])
        if args.environment is not None:
            lines.append(f'source {args.environment}/bin/activate')

        config_updates_commands = []
        stack = config_updates
        while stack:
            key, value = stack.popitem()
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    stack[f'{key}.{sub_key}'] = sub_value
            else:
                config_updates_commands.append(f'{key}={value}')

        command = f'python -m attack_evaluation.run -F {result_dir} with {" ".join(named_configs)} {" ".join(config_updates_commands)}'
        lines.append(command)

        job_file = slurm_script_dir / f'{Path(args.config).stem}_{i:04d}.job'
        with open(job_file, 'w') as fh:
            fh.write('\n'.join(lines))

        if args.submit:
            os.system(f'sbatch {job_file}')
