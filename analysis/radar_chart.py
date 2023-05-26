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

threat_model_labels = {
    'l0': r'$\ell_0$',
    'l1': r'$\ell_1$',
    'l2': r'$\ell_2$',
    'linf': r'$\ell_{\infty}$',
}

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

    attack_models_optimality = defaultdict(dict)
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

            attack_models_optimality[attack_label][scenario.model] = optimality

    for attack_label in attack_models_optimality.keys():
        print(attack_models_optimality[attack_label])

        labels = list(attack_models_optimality[attack_label].keys())
        markers = np.linspace(0, 1, 5)


        def make_radar_chart(attack_label, stats, attribute_labels=labels, plot_markers=markers):
            labels = np.array(attribute_labels)

            angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
            stats = np.concatenate((stats, [stats[0]]))
            angles = np.concatenate((angles, [angles[0]]))
            labels = np.concatenate((labels, [labels[0]]))

            fig = plt.figure()
            ax = fig.add_subplot(111, polar=True)
            ax.plot(angles, stats, 'o-', linewidth=3)
            ax.fill(angles, stats, alpha=0.25)
            ax.set_thetagrids(angles * 180 / np.pi, labels, fontsize=13)
            ax.legend(loc='center right', labelspacing=.1, handletextpad=0.5)
            fig_name = result_path / f"attack_{attack_label}.pdf"
            plt.yticks(plot_markers)
            ax.set_title(attack_label, pad=10)
            ax.grid(True)
            fig.savefig(fig_name, bbox_inches='tight')
            plt.show()

            return plt.show()

        make_radar_chart(attack_label, list(attack_models_optimality[attack_label].values()))
