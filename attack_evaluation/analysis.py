# %%
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.ticker as ticker
from sacred import Experiment
import torch
import numpy as np
import seaborn as sns

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 3})
sns.despine(left=True)

ex = Experiment('attack_evaluation_curves')

_eval_distances = {'l2': '\ell_2'}


@ex.config
def config():
    dataset = 'mnist'
    model = 'mnist_smallcnn'
    attacks = 'fmn,alma,ddn'
    norm = 'l2'
    exp_id = 1


@ex.automain
def main(dataset, model, attacks, norm, exp_id, _config, _run, _log):
    for dist_key in _eval_distances.keys():

        fig, ax = plt.subplots(figsize=(5, 4))
        for attack in attacks.split(','):
            root_dir = Path(f'{dataset}-{model}-{attack}-{norm}') / Path(f'{exp_id}')
            load_dir = root_dir / f'attack_data.pt'

            attack_data = torch.load(load_dir)

            perturbation_size = np.array(attack_data['distances'][dist_key])
            idx_sorted = np.argsort(perturbation_size)
            distances = perturbation_size[idx_sorted]

            ASR_values = np.array(attack_data['adv_success'])
            y_axis = ASR_values[idx_sorted].cumsum()
            robust_acc = 1 - y_axis / len(y_axis)

            cnt = sns.lineplot(x=distances, y=robust_acc, ax=ax, linestyle='--',
                               label=f'{attack} $%s$' % _eval_distances[dist_key])
            cnt.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
            cnt.xaxis.set_major_locator(ticker.MultipleLocator(0.5))

        ax.set_ylabel('Robust Accuracy')
        ax.set_xlabel('Perturbation Size $%s$' % _eval_distances[dist_key])
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.savefig(f'{dataset}-{model}-{norm}-{exp_id}-rbst_curves.pdf', bbox_inches='tight')
        plt.show()
