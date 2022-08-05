import argparse
import json
import pathlib

import numpy as np
from matplotlib import pyplot as plt, ticker

from read import read_results

parser = argparse.ArgumentParser('Plot results')

parser.add_argument('--dir', '-d', type=str, default='results', help='Directory used to store experiment results')
parser.add_argument('--dataset', type=str, default=None, help='Dataset for which to plot results')
parser.add_argument('--model', '-m', type=str, default=None, help='Model for which to plot results')
parser.add_argument('--threat-model', '--tm', type=str, default=None, help='Threat model for which to plot results')

args = parser.parse_args()

# check that result directory exists
result_path = pathlib.Path(args.dir)
assert result_path.exists()

# find applicable scenarios
dataset = '*' or args.dataset
model = '*' or args.model
threat_model = '*' or args.threat_model
scenario_pattern = '-'.join((dataset, model, threat_model)) + '.json'

best_distances_files = result_path.glob(scenario_pattern)

for best_distances_file in best_distances_files:
    scenario_stem = best_distances_file.relative_to(result_path).stem
    scenario = scenario_stem.split('-')
    assert len(scenario) == 3, f'Best distance file {best_distances_file} saved with more than 3 "-" in name.'

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

    # find results corresponding to scenario
    scenario_path = result_path.joinpath(*scenario)
    info_files = scenario_path.glob('**/**/info.json')

    for info_file in sorted(info_files):
        hash_distances = read_results(info_file=info_file)[1]

        attack_folder = info_file.parent.relative_to(scenario_path).as_posix()
        adv_distances = np.array(list(hash_distances.values()))

        distances, counts = np.unique(adv_distances, return_counts=True)
        robust_acc = 1 - counts.cumsum() / len(adv_distances)

        distances_clipped, counts = np.unique(adv_distances.clip(min=None, max=max_dist), return_counts=True)
        robust_acc_clipped = 1 - counts.cumsum() / len(adv_distances)

        area = np.trapz(robust_acc_clipped, distances_clipped)
        optimality = 1 - (area - best_area) / (clean_acc * max_dist - best_area)

        ax.plot(distances, robust_acc, linestyle='--', label=f'{attack_folder}: {optimality:.2%}')

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
    fig_name = f'{scenario_stem}.pdf'
    fig.savefig(result_path / fig_name, bbox_inches='tight')
