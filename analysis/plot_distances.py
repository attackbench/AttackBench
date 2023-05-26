import argparse
import json
import os
import pathlib
import warnings
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt, ticker

from compile import compile_scenario
from read import read_distances
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
        model = args.model or '*'
        library = f'{args.library}_*/**' if args.library else '**'
        batch_size = f'batch_size_{args.batch_size}' if args.batch_size else '*'
        info_files_paths = os.sep.join((dataset, threat_model, model, batch_size, library, 'info.json'))
        info_files = result_path.glob(info_files_paths)

    for info_file in info_files:
        scenario, hash_distances = read_distances(info_file, distance_type=distance_type)
        to_plot[scenario].append((info_file.parent, hash_distances))

    for scenario in to_plot.keys():
        print(scenario)
        best_distances_file = result_path / f'{scenario.dataset}-{scenario.threat_model}-{scenario.model}-{scenario.batch_size}.json'
        if not best_distances_file.exists():
            print("Compiling ", scenario)
            warnings.warn(f'Best distances files {best_distances_file} does not exist for scenario {scenario}.')
            warnings.warn(f'Compiling best distances file for scenario {scenario}')
            compile_scenario(path=result_path, scenario=scenario, distance_type=distance_type)

        with open(best_distances_file, 'r') as f:
            data = json.load(f)
        best_distances = list(data.values())

        # plot best distances
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(5, 4), layout='constrained')
        ax.set_title(' - '.join(scenario), pad=10)

        distances, counts = np.unique(best_distances, return_counts=True)
        robust_acc = 1 - counts.cumsum() / len(best_distances)
        ax.plot(distances, robust_acc, linestyle='-', label=f'Best distances', c='k', linewidth=1)

        # get quantities for optimality calculation
        clean_acc = np.count_nonzero(best_distances) / len(best_distances)
        max_dist = np.amax(distances)
        best_area = np.trapz(robust_acc, distances)
        plot_xlim = max_dist * 1.5

        attacks_to_plot = {}
        for attack_folder, hash_distances in sorted(to_plot[scenario]):
            adv_distances = np.array(list(hash_distances.values()))
            distances, counts = np.unique(adv_distances, return_counts=True)
            robust_acc = 1 - counts.cumsum() / len(adv_distances)

            distances_clipped, counts = np.unique(adv_distances.clip(min=None, max=max_dist), return_counts=True)
            robust_acc_clipped = 1 - counts.cumsum() / len(adv_distances)

            if distances[-1] == np.inf:
                distances[-1] = plot_xlim
                robust_acc[-1] = robust_acc[-2]
                robust_acc_clipped[-1] = robust_acc_clipped[-2]

            area = np.trapz(robust_acc_clipped, distances_clipped)
            optimality = 1 - (area - best_area) / (clean_acc * max_dist - best_area)

            attack_label = attack_folder.relative_to(attack_folder.parents[1]).as_posix()
            attacks_to_plot[attack_label] = {'distances': distances, 'area': area, 'robust_acc': robust_acc,
                                             'optimality': optimality}

            # ax.plot(distances, robust_acc, linewidth=1, linestyle='--', label=f'{attack_label}: {optimality:.2%}')

        for attack_label in top_k_attacks(attacks_to_plot, k=args.K):
            atk = attacks_to_plot[attack_label]
            ax.plot(atk['distances'], atk['robust_acc'], linewidth=1, linestyle='--',
                    label=f'{attack_label}: {atk["optimality"]:.2%}')

        ax.grid(True, linestyle='--', c='lightgray', which='major')
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(1))
        ax.set_xlim(left=0, right=plot_xlim)
        ax.set_ylim(bottom=0, top=1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_ylabel('Robust Accuracy (%)')
        ax.set_xlabel(f'Perturbation Size {threat_model_labels[scenario.threat_model]}')

        ax.annotate(text=f'Clean accuracy: {clean_acc:.2%}', xy=(0, clean_acc),
                    xytext=(ax.get_xlim()[1] / 2, clean_acc), ha='left', va='center',
                    arrowprops={'arrowstyle': '-', 'linestyle': '--'})

        ax.legend(loc='center right', labelspacing=.1, handletextpad=0.5)

        library = args.library or "all"
        parts = [scenario.dataset, scenario.threat_model, scenario.model, scenario.batch_size, library]
        if args.suffix:
            parts.append(args.suffix)
        fig_name = result_path / f'{"-".join(parts)}.pdf'
        fig.savefig(fig_name)
        fig.show()
        plt.close()
