import argparse
import json
import os
import pathlib
import warnings
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt, ticker

from compile import compile_scenario
from read import read_results

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Plot results')

    parser.add_argument('--dir', '-d', type=str, default='results', help='Directory used to store experiment results')
    parser.add_argument('--dataset', type=str, default=None, help='Dataset for which to plot results')
    parser.add_argument('--threat-model', '--tm', type=str, default=None, help='Threat model for which to plot results')
    parser.add_argument('--model', '-m', type=str, default=None, help='Model for which to plot results')
    parser.add_argument('--info-files', '--if', type=str, nargs='+', default=None,
                        help='List of info files to plot from.')

    args = parser.parse_args()

    # check that result directory exists
    result_path = pathlib.Path(args.dir)
    assert result_path.exists()

    to_plot = defaultdict(list)
    if args.info_files is not None:
        info_files = [pathlib.Path(info_file) for info_file in args.info_files]
        assert all(info_file.exists() for info_file in info_files)
    else:
        dataset = args.dataset or '*'
        threat_model = args.threat_model or '*'
        model = args.model or '*'
        info_files = result_path.glob(os.sep.join((dataset, threat_model, model, '**', 'info.json')))

    for info_file in info_files:
        scenario, hash_distances = read_results(info_file)
        to_plot[scenario].append((info_file.parent, hash_distances))

    for scenario in to_plot.keys():
        best_distances_file = result_path / f'{scenario.dataset}-{scenario.threat_model}-{scenario.model}.json'
        if not best_distances_file.exists():
            warnings.warn(f'Best distances files {best_distances_file} does not exist for scenario {scenario}.')
            warnings.warn(f'Compiling best distances file for scenario {scenario}')
            compile_scenario(path=result_path, scenario=scenario)

        with open(best_distances_file, 'r') as f:
            data = json.load(f)
        best_distances = [d[1] for d in data.values()]

        # plot best distances
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.set_title(' - '.join(scenario), pad=10)

        distances, counts = np.unique(best_distances, return_counts=True)
        robust_acc = 1 - counts.cumsum() / len(best_distances)
        ax.plot(distances, robust_acc, linestyle='-', label=f'Best distances', c='k', linewidth=1)

        # get quantities for optimality calculation
        clean_acc = np.count_nonzero(best_distances) / len(best_distances)
        max_dist = np.amax(distances)
        best_area = np.trapz(robust_acc, distances)

        for attack_folder, hash_distances in sorted(to_plot[scenario]):
            adv_distances = np.array(list(hash_distances.values()))
            distances, counts = np.unique(adv_distances, return_counts=True)
            robust_acc = 1 - counts.cumsum() / len(adv_distances)

            distances_clipped, counts = np.unique(adv_distances.clip(min=None, max=max_dist), return_counts=True)
            robust_acc_clipped = 1 - counts.cumsum() / len(adv_distances)

            area = np.trapz(robust_acc_clipped, distances_clipped)
            optimality = 1 - (area - best_area) / (clean_acc * max_dist - best_area)

            attack_label = attack_folder.relative_to(attack_folder.parents[1]).as_posix()
            ax.plot(distances, robust_acc, linestyle='--', label=f'{attack_label}: {optimality:.2%}')

        ax.grid(True, linestyle='--', c='lightgray', which='major')
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(1))
        ax.set_xlim(left=0, right=max_dist * 1.05)
        ax.set_ylim(bottom=0, top=1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_ylabel('Robust Accuracy (%)')
        ax.set_xlabel(f'Perturbation Size {scenario[-1]}')

        ax.annotate(text=f'Clean accuracy: {clean_acc:.2%}', xy=(0, clean_acc),
                    xytext=(ax.get_xlim()[1] / 2, clean_acc), ha='left', va='center',
                    arrowprops={'arrowstyle': '-', 'linestyle': '--'})

        ax.legend(loc='center right', labelspacing=.1, handletextpad=0.5)
        fig.tight_layout()
        fig_name = result_path / f'{scenario.dataset}-{scenario.threat_model}-{scenario.model}.pdf'
        fig.savefig(fig_name, bbox_inches='tight')