import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from sacred import Experiment

ex = Experiment('attack_evaluation_curves')

_eval_distances = {'l2': '\ell_2'}

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

            config_file = attack_dir / f'config.json'
            with open(config_file, 'r') as f:
                config = json.load(f)

            adv_distances = np.array(attack_data['distances'][dist_key])
            success = np.array(attack_data['adv_success'])
            distances, counts = np.unique(adv_distances[success], return_counts=True)

            robust_acc = 1 - counts.cumsum() / len(success)
            # compute the area under the curve for now => will need to replace that by the sub-optimality metric later
            curve_area = np.trapz(robust_acc, distances)

            ax.plot(distances, robust_acc, linestyle='--',
                    label=f"{config['attack']['name']} ${_eval_distances[dist_key]}$ {curve_area:.3f}")

        ax.grid(True, linestyle='--', c='lightgray', which='major')
        ax.yaxis.set_major_formatter(ticker.PercentFormatter(1))
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0, top=1)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        ax.set_ylabel('Robust Accuracy (%)')
        ax.set_xlabel(f'Perturbation Size ${_eval_distances[dist_key]}$')

        ax.annotate(text=f'Clean accuracy: {attack_data["accuracy"]:.2%}', xy=(0, attack_data['accuracy']),
                    xytext=(ax.get_xlim()[1] / 2, attack_data['accuracy']), ha='left', va='center',
                    arrowprops={'arrowstyle': '-', 'linestyle': '--'})

        ax.legend(loc='center right', labelspacing=.1, handletextpad=0.5)
        fig.tight_layout()
        fig_name = f"{config['model']['name']}-{config['attack']['norm']}-{exp_id}-rbst_curves_{dist_key}.pdf"
        fig.savefig(fig_path / fig_name, bbox_inches='tight')
        plt.show()
