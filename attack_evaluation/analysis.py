import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from sacred import Experiment

ex = Experiment('attack_evaluation_curves')

_eval_distances = {'l2': '\ell_2'}
eps_threshold = 1e-02


@ex.config
def config():
    root = 'experimental_results'
    dataset = 'mnist'
    model = 'mnist_smallcnn'
    attacks = 'fmn,alma,ddn'
    norm = 'l2'
    exp_id = 1


@ex.automain
def main(root, dataset, model, attacks, norm, exp_id, _config, _run, _log):
    exp_dir = Path(root) / dataset
    fig_path = exp_dir / 'figs'

    for dist_key in _eval_distances.keys():
        fig, ax = plt.subplots(figsize=(5, 4))
        for attack in attacks.split(','):
            attack_dir = exp_dir / f'{dataset}-{model}-{attack}-{norm}' / f'{exp_id}'
            info_file = attack_dir / f'info.json'

            with open(info_file, 'r') as f:
                attack_data = json.load(f)
            fig_path.mkdir(exist_ok=True)

            adv_distances = np.array(attack_data['distances'][dist_key])
            success = np.array(attack_data['adv_success'])
            adv_distances[~success] = float('inf')
            distances = np.sort(np.unique(adv_distances[success]))

            if 0 not in distances:
                distances = np.insert(distances, 0, 0)

            robust_acc = (adv_distances[None, :] > distances[:, None]).mean(axis=1)
            curve_area = np.trapz(robust_acc, distances)

            ax.plot(distances, robust_acc, linestyle='--',
                    label=f'{attack} ${_eval_distances[dist_key]}$ {curve_area:.2f}')

        ax.grid(True, linestyle='--', c='lightgray', which='major')
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(1))
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0, top=1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_ylabel('Robust Accuracy (%)')
        ax.set_xlabel(f'Perturbation Size ${_eval_distances[dist_key]}$')

        ax.axhline(attack_data['accuracy'], linestyle='--', color='black', linewidth=1)
        ax.annotate(text=f'Clean accuracy: {attack_data["accuracy"]:.2%}', xy=(0.5, attack_data['accuracy']),
                    xycoords='axes fraction', xytext=(0, -3), textcoords='offset points',
                    horizontalalignment='center', verticalalignment='top')

        plt.legend(loc='center right', labelspacing=.1, handletextpad=0.5)
        plt.tight_layout()
        plt.savefig(fig_path / f'{model}-{norm}-{exp_id}-rbst_curves_{dist_key}.pdf', bbox_inches='tight')
        plt.show()
