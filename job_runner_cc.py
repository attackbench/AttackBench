import argparse
import json
import math
import os
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
    parser.add_argument('--models', type=str, default='all', nargs='+',
                        choices=available_models + ['all'], help='Victim model')
    parser.add_argument('--threat-models', type=str, default='all', nargs='+',
                        choices=available_threat_models + ['all'], help='Threat model for which to run the attacks')
    parser.add_argument('--libraries', '--lib', type=str, default='all', nargs='+',
                        choices=available_libraries + ['all'], help='Attack library')
    parser.add_argument('--attacks', type=str, default='all', nargs='+',
                        choices=available_attacks + ['all'], help='Attacks to run')
    parser.add_argument('--batch-size', type=int, default=128, help='Batch size')
    parser.add_argument('--num-samples', type=int, default=None, help='Number of samples for SubDataset.')
    parser.add_argument('--seed', type=int, default=42, help='Seed for the experiments')

    args = parser.parse_args()

    # exp setup
    exp_dir = Path(os.path.dirname(os.path.realpath(__file__)))
    result_dir = Path(args.result_dir)

    # replace 'all' args
    potential_all_args = ['models', 'threat_models', 'libraries', 'attacks']
    available_all_args = [available_models, available_threat_models, available_libraries, available_attacks]
    for arg_name, available_args in zip(potential_all_args, available_all_args):
        if 'all' in getattr(args, arg_name):
            setattr(args, arg_name, available_args)

    combinations = product(args.models, args.threat_models, args.libraries, args.attacks)
    for (model, threat_model, library, attack) in combinations:

        attack_named_config = attack_configs[threat_model][library]
        attack_name = f'{attack_named_config["prefix"]}_{attack}'

        if attack not in attack_configs[threat_model][library]['attacks']:
            continue  # skip if attack is not available for threat model and library

        dataset = ex.ingredients[1].named_configs[model]()['dataset']
        result_path = result_dir / dataset / threat_model / model / f'batch_size_{args.batch_size}' / attack_name
        result_path.mkdir(parents=True, exist_ok=True)

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
            named_configs = (f'model.{model}', f'attack.{attack_name}')
            config_updates = {'attack': {'threat_model': threat_model},
                              'dataset': {'batch_size': args.batch_size, 'num_samples': 3 * args.batch_size}}
            try:
                run = ex.run(config_updates=config_updates, named_configs=named_configs,
                             options={'--loglevel': 'ERROR'})
            except:
                print(f'Skipping {threat_model} | {model} | {library} | {attack} (crashed).')
                continue

            times = run.info['times']
            time_per_batch = np.median(times)
            num_samples = dataset_lengths[dataset] if args.num_samples is not None else dataset_lengths[dataset]
            num_batches = math.ceil(min(num_samples, dataset_lengths[dataset]) / args.batch_size)
            total_time = max(args.min_time, math.ceil(time_per_batch * num_batches * 1.1))  # add 10%
            hours, minutes, seconds = total_time // 3600, (total_time % 3600) // 60, total_time % 60
            time_string = f'{hours:02d}:{minutes:02d}:{seconds:02d}'
            print(f'Running {threat_model} | {model} | {library} | {attack} for {total_time}s = {time_string}')
            lines.append(f'#SBATCH --time={time_string}')
        else:
            lines.append(f'#SBATCH --time={args.time}')

        lines.extend(['module load python/3.9', f'cd {exp_dir.as_posix()}'])
        if args.environment is not None:
            lines.append(f'source {args.environment}/bin/activate')

        job_file = result_path / f'{threat_model}-{model}-{library}-{attack}.job'
        command = f'python -m attack_evaluation.run -F {result_dir} with ' \
                  f'seed={args.seed} ' \
                  f'dataset.num_samples={args.num_samples} ' \
                  f'dataset.batch_size={args.batch_size} ' \
                  f'model.{model} ' \
                  f'attack.{attack_name} ' \
                  f'attack.threat_model={threat_model}'
        lines.append(command)

        with open(job_file, 'w') as fh:
            fh.write('\n'.join(lines))

        if args.submit:
            os.system(f'sbatch {job_file}')
