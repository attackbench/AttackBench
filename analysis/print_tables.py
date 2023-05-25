import argparse
import json
import os
import pathlib
import warnings
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt, ticker

from compile import compile_scenario
from read import read_distances, read_info
from utils import top_k_attacks

from tabulate import tabulate

threat_model_labels = {
    'l0': r'$\ell_0$',
    'l1': r'$\ell_1$',
    'l2': r'$\ell_2$',
    'linf': r'$\ell_{\infty}$',
}

ROUND = lambda x: np.around(x, 3)
TOLERANCE = 1e-06

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Plot results')

    parser.add_argument('--dir', '-d', type=str, default='results', help='Directory used to store experiment results')
    parser.add_argument('--dataset', type=str, default=None, help='Dataset for which to plot results')
    parser.add_argument('--threat-model', '--tm', type=str, default=None, help='Threat model for which to plot results')
    parser.add_argument('--model', '-m', type=str, default=None, help='Model for which to plot results')
    parser.add_argument('--library', '-l', type=str, default=None, help='Library for which to plot results')
    parser.add_argument('--K', '-k', type=int, default=None, help='Top K attacks to show')
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

    Table = {}
    for key in threat_model_lst:
        Table[key] = [
            ['Dataset', 'BatchSize', 'Threat', 'Model', 'Attack', '$\ell_p$', '$\ell_p^\star$', 'ASR', 'Optimality',
             '#Forwards', '#Backwards',
             'hasBoxFailure', 'isOutputOptimal']]
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

        distances, counts = np.unique(best_distances, return_counts=True)
        robust_acc = 1 - counts.cumsum() / len(best_distances)

        # get quantities for optimality calculation
        clean_acc = np.count_nonzero(best_distances) / len(best_distances)
        max_dist = np.amax(distances)
        best_area = np.trapz(robust_acc, distances)

        attacks_to_plot = {}
        for attack_folder, info in sorted(to_plot[scenario]):
            adv_distances = info['best_optim_distances'] if distance_type == 'best' else info['distances']
            distances, counts = np.unique(adv_distances, return_counts=True)
            robust_acc = 1 - counts.cumsum() / len(adv_distances)

            distances_clipped, counts = np.unique(adv_distances.clip(min=None, max=max_dist), return_counts=True)
            robust_acc_clipped = 1 - counts.cumsum() / len(adv_distances)

            area = np.trapz(robust_acc_clipped, distances_clipped)
            optimality = 1 - (area - best_area) / (clean_acc * max_dist - best_area)

            attack_label = attack_folder.relative_to(attack_folder.parents[1]).as_posix()
            attack_label = attack_label.split('/')[0]

            adv_valid_success = info['adv_valid_success']
            median_dist = np.median(info['distances'][adv_valid_success])
            median_best_dist = np.median(info['distances'][adv_valid_success])
            is_dist_optimal = np.isclose(info['best_optim_distances'], info['distances'], rtol=TOLERANCE).all()

            ASR = info['ASR']
            avg_n_forwards = int(np.mean(info['num_forwards']))
            avg_n_backwards = int(np.mean(info['num_backwards']))
            has_box_failures = np.any(info['box_failures'])
            row = list(scenario) + [attack_label, ROUND(median_dist), ROUND(median_best_dist), ROUND(ASR), ROUND(optimality),
                                    avg_n_forwards, avg_n_backwards, has_box_failures, is_dist_optimal]
            attacks_to_plot[attack_label] = {'row': row, 'optimality': optimality, 'area': area}

        for attack_label in top_k_attacks(attacks_to_plot, k=args.K):
            atk = attacks_to_plot[attack_label]
            Table[scenario.threat_model].append(atk['row'])

    for key in threat_model_lst:
        np.savetxt(f"output_{key}.csv", Table[key], delimiter=",", fmt='%s')
        print(tabulate(Table[key], headers="firstrow", missingval="-", tablefmt="rst", floatfmt="0.2f"))
        print()