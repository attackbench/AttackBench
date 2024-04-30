import argparse
import json
import os
import pathlib
import warnings
from collections import defaultdict
from scipy.stats import wilcoxon, mannwhitneyu, ks_2samp
from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt, ticker
from itertools import combinations
from compile import compile_scenario
from read import read_distances, read_info
from utils import top_k_attacks, eval_optimality, ensemble_distances
import pandas as pd
from tabulate import tabulate
from itertools import product
from utils import rename_attack

threat_model_labels = {
    'l0': r'$\ell_0$',
    'l1': r'$\ell_1$',
    'l2': r'$\ell_2$',
    'linf': r'$\ell_{\infty}$',
}

ROUND = lambda x: np.around(x, 4)
TOLERANCE = 1e-04
TEST_SIGNIFICANCE = 0.01

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Plot results')

    parser.add_argument('--dir', '-d', type=str, default='results', help='Directory used to store experiment results')
    parser.add_argument('--cdir', '-cd', type=str, default='compiled', help='Directory used to store compiled results')
    parser.add_argument('--dataset', type=str, default=None, help='Dataset for which to plot results')
    parser.add_argument('--threat-model', '--tm', type=str, default=None, help='Threat model for which to plot results')
    parser.add_argument('--model', '-m', type=str, default=None, help='Model for which to plot results')
    parser.add_argument('--library', '-l', type=str, default=None, help='Library for which to plot results')
    parser.add_argument('--K', '-k', type=int, default=None, help='Top K attacks to show')
    parser.add_argument('--LE', '-le', type=int, default=1, help='Levels of ensemble')
    parser.add_argument('--KE', '-ke', type=int, default=1, help='Top K ensembles for next level')
    parser.add_argument('--batch_size', '-bs', type=int, default=None, help="Batch size for which to plot the results")
    parser.add_argument('--info-files', '--if', type=str, nargs='+', default=None,
                        help='List of info files to plot from.')
    parser.add_argument('--distance_type', '-dist', type=str, default='best', choices=['best', 'actual'],
                        help='Define distances to plot results')
    parser.add_argument('--suffix', '-s', type=str, default=None, help='Suffix for the name of the plot')

    args = parser.parse_args()

    # check that result directory exists
    result_path = pathlib.Path(args.dir)
    assert result_path.exists()

    distance_type = args.distance_type

    to_plot = defaultdict(list)
    if args.info_files is not None:
        info_files_paths = args.info_files
        info_files = [pathlib.Path(info_file) for info_file in info_files_paths]
        assert all(info_file.exists() for info_file in info_files)
    else:
        dataset = args.dataset or '*'
        threat_model = args.threat_model or '*'
        threat_model_lst = threat_model_labels.keys() if threat_model == '*' else [threat_model]
        model = args.model or '*'
        library = f'{args.library}_*/**' if args.library else '**'
        batch_size = f'batch_size_{args.batch_size}' if args.batch_size else '*'
        info_files_paths = os.sep.join((dataset, threat_model, model, batch_size, library, 'info.json'))
        info_files = result_path.glob(info_files_paths)

    for info_file in info_files:
        scenario, info = read_info(info_file)
        to_plot[scenario].append((info_file.parent, info))

    df = pd.DataFrame()
    for scenario in to_plot.keys():
        best_distances_file = result_path / f'{scenario.dataset}-{scenario.threat_model}-{scenario.model}-{scenario.batch_size}.json'
        if not best_distances_file.exists():
            # print("Compiling ", scenario)
            warnings.warn(f'Best distances files {best_distances_file} does not exist for scenario {scenario}.')
            warnings.warn(f'Compiling best distances file for scenario {scenario}')
            compile_scenario(path=result_path, scenario=scenario, distance_type=distance_type)

        with open(best_distances_file, 'r') as f:
            data = json.load(f)
        best_distances = list(data.values())

        attacks_to_plot = {}
        for attack_folder, info in sorted(to_plot[scenario]):
            adv_distances = info['best_optim_distances'] if distance_type == 'best' else info['distances']
            optimality = eval_optimality(adv_distances=adv_distances, best_distances=best_distances)

            attack_label = attack_folder.relative_to(attack_folder.parents[1]).as_posix()

            avg_n_forwards = int(np.mean(info['num_forwards']))
            avg_n_backwards = int(np.mean(info['num_backwards']))
            avg_exec_time = np.mean(info['times'])

            library, attack = rename_attack(attack_label).split(' ', 1)
            df = df.append({
                'dataset': scenario.dataset,
                'attack': attack,
                'library': library,
                'model': scenario.model,
                'threat_model': scenario.threat_model,
                'ASR': info['ASR'],
                'optimality': optimality,
                'n_forwards': avg_n_forwards,
                'n_backwards': avg_n_backwards,
                'time': avg_exec_time
            }, ignore_index=True)

    for (dataset, model, threat), sub_df in df.groupby(['dataset', 'model', 'threat_model']):
        compile_dir = Path(args.cdir or 'compiled') / Path(dataset)
        compile_dir.mkdir(exist_ok=True, parents=True)
        sub_df.to_json(path_or_buf=compile_dir/f'{model}_{threat}.json', orient='records')