# %%
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.ticker as ticker
from sacred import Experiment
import torch
import numpy as np
import seaborn as sns
from utils import mkdir_p

sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 3})
sns.despine(left=True)

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


def area(x, y):
    p = x[1:] - x[:-1]
    k = y[1:] + y[:-1]
    return 1 - (p @ k) / (2 * x.max())

@ex.automain
def main(root, dataset, model, attacks, norm, exp_id, _config, _run, _log):
    exp_dir = Path(root) / dataset
    fig_path = exp_dir / 'figs'

    for dist_key in _eval_distances.keys():
        j = 1
        fig, ax = plt.subplots(figsize=(5, 4))
        for attack in attacks.split(','):
            attack_dir = exp_dir / f'{dataset}-{model}-{attack}-{norm}' / Path(f'{exp_id}')
            filename = attack_dir / f'attack_data.pt'

            attack_data = torch.load(filename)
            mkdir_p(fig_path)

            perturbation_size = np.array(attack_data['distances'][dist_key])
            idx_sorted = np.argsort(perturbation_size)
            distances = perturbation_size[idx_sorted]

            ASR_values = np.array(attack_data['adv_success'])
            y_axis = ASR_values[idx_sorted].cumsum()
            robust_acc = 1 - y_axis / len(y_axis)

            curve_area = area(distances, robust_acc) #1 - (robust_acc.sum() / len(robust_acc))

            cnt = sns.lineplot(x=distances, y=robust_acc, ax=ax, linestyle='--',
                               label=f'{attack} $%s$ %.2f' % (_eval_distances[dist_key], curve_area))

            lower_thr = np.where(robust_acc < eps_threshold)[0]
            if len(lower_thr) > 0:
                eps_0 = distances[lower_thr][0]
                c = ax.get_lines()[-1].get_c()
                plt.axvline(eps_0, -0.1, 0.1, color=c, linewidth=1)
                plt.text(eps_0, 0.1 * j, "$%s_{%s}$" % ('\epsilon', '0\%'), horizontalalignment='center',
                         size='small', color=c)
                j *= -1
            cnt.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
            # cnt.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
            cnt.axhline(attack_data['accuracy'], linestyle="--", color="black", linewidth=1)

        ax.set_ylabel('Robust Accuracy')
        ax.set_xlabel('Perturbation Size $%s$' % _eval_distances[dist_key])
        plt.text(distances.mean(), attack_data['accuracy'] + 0.01, "Clean accuracy",
                 horizontalalignment='left', size='small', color='black', weight='normal')

        plt.legend(loc='center right', labelspacing=.1, handletextpad=0.5)
        plt.tight_layout()
        plt.savefig(fig_path / f'{model}-{norm}-{exp_id}-rbst_curves_{dist_key}.pdf', bbox_inches='tight')
        plt.show()
