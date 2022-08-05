import argparse
import itertools
import json
import pathlib
import warnings
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt, ticker

parser = argparse.ArgumentParser('Compile results from several attacks')

parser.add_argument('--dir', '-d', type=str, default='results', help='Directory used to store experiment results')
parser.add_argument('--plot', action='store_true', help='Plot all results')

args = parser.parse_args()

# check that result directory exists
result_path = pathlib.Path(args.dir)
assert result_path.exists()

# find info files corresponding to finished experiments
info_files = result_path.glob('**/info.json')

results = defaultdict(list)

# traverse result files
for info_file in info_files:
    # read config file
    with open(info_file.parent / 'config.json', 'r') as f:
        config = json.load(f)

    # extract main configs
    dataset = config['dataset']['name']
    model = config['model']['name']
    threat_model = config['attack']['threat_model']

    # read info file
    with open(info_file, 'r') as f:
        info = json.load(f)

    # get hashes and distances for the adversarial examples
    hashes = info['hashes']
    distances = np.array(info['distances'][threat_model])
    ori_success = np.array(info['ori_success'])
    adv_success = np.array(info['adv_success'])

    # check that adversarial examples have 0 distance for adversarial clean samples
    non_zero_ori_success = any([d != 0 for d in itertools.compress(distances, ori_success)])
    if (n := np.count_nonzero(distances[ori_success])):
        warnings.warn(f'{n} already adversarial clean samples have non zero perturbations.')

    # replace distances with inf for failed attacks and 0 for already adv clean samples
    distances[~adv_success] = float('inf')
    distances[ori_success] = 0

    # store results
    scenario = (dataset, model, threat_model)
    results[scenario].append((info_file.parent, {hash: distance for (hash, distance) in zip(hashes, distances)}))

# compile and save best distances
for scenario in results.keys():
    best_distances = {}

    for attack_dir, hash_distances in results[scenario]:
        attack_path = attack_dir.relative_to(result_path)
        for hash, distance in hash_distances.items():
            best_distance = best_distances.get(hash, (None, float('inf')))[1]
            if distance < best_distance:  # TODO: fix ambiguous case where two attacks have the same best distance
                best_distances[hash] = (attack_path.as_posix(), distance)

    with open(result_path / f'{"-".join(scenario)}.json', 'w') as f:
        json.dump(best_distances, f)

if args.plot:
    for scenario in results.keys():

        with open(result_path / f'{"-".join(scenario)}.json', 'r') as f:
            data = json.load(f)
        best_distances = [d[1] for d in data.values()]

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.set_title(' - '.join(scenario), pad=10)

        distances, counts = np.unique(best_distances, return_counts=True)
        robust_acc = 1 - counts.cumsum() / len(best_distances)
        ax.plot(distances, robust_acc, linestyle='-', label=f'Best distances', c='k', linewidth=1)

        clean_acc = np.count_nonzero(best_distances) / len(best_distances)
        max_dist = np.amax(distances)
        best_area = np.trapz(robust_acc, distances)

        for attack_dir, hash_distances in sorted(results[scenario]):
            attack_folder = attack_dir.relative_to(attack_dir.parent.parent)
            adv_distances = np.array(list(hash_distances.values()))

            distances, counts = np.unique(adv_distances, return_counts=True)
            robust_acc = 1 - counts.cumsum() / len(adv_distances)

            distances_clipped, counts = np.unique(adv_distances.clip(min=None, max=max_dist), return_counts=True)
            robust_acc_clipped = 1 - counts.cumsum() / len(adv_distances)

            area = np.trapz(robust_acc_clipped, distances_clipped)
            optimality = 1 - (area - best_area) / (clean_acc * max_dist - best_area)

            ax.plot(distances, robust_acc, linestyle='--', label=f'{attack_folder.as_posix()}: {optimality:.2%}')

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
        fig_name = f'{"-".join(scenario)}.pdf'
        fig.savefig(result_path / fig_name, bbox_inches='tight')
