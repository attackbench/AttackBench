import argparse
import json
import os
import pathlib
import warnings
from collections import defaultdict

import numpy as np
from matplotlib import pyplot as plt, ticker

from read import read_results
from utils import ensemble_gain, _MAX_GAIN

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
    parser.add_argument('--batch_size', '-bs', type=int, default=None, help="Batch size for which to plot the results")
    parser.add_argument('--info-files', '--if', type=str, nargs='+', default=None,
                        help='List of info files to plot from.')
    parser.add_argument('--suffix', '-s', type=str, default=None, help='Suffix for the name of the plot')

    args = parser.parse_args()

    # check that result directory exists
    result_path = pathlib.Path(args.dir)
    assert result_path.exists()

    to_plot = defaultdict(dict)
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
        scenario, info_attack = read_results(info_file)
        to_plot[scenario] = info_attack

    atks_gain = np.zeros((len(to_plot), len(to_plot)))

    for i, scenario_atk1 in enumerate(to_plot.keys()):
        atk1_info = to_plot[scenario_atk1]
        for j, scenario_atk2 in enumerate(to_plot.keys()):
            atk2_info = to_plot[scenario_atk2]
            atks_gain[i, j] = ensemble_gain(atk1_info['adv_success'], atk2_info['adv_success'])
            atks_gain[j, i] = ensemble_gain(atk2_info['adv_success'], atk1_info['adv_success'])

    fig, ax = plt.subplots(figsize=(7, 7), layout="constrained")
    #ax.set_title(f'{args.library} ensemble gain', pad=5)

    ax.matshow(atks_gain.transpose(), origin='upper', cmap=plt.cm.Blues, alpha=0.75)
    for i in range(len(atks_gain)):
        for j in range(len(atks_gain)):
            c = '$\infty$' if atks_gain[i, j] == _MAX_GAIN else '%.3f' % atks_gain[i, j]
            ax.text(i, j, c, va='center', ha='center')

    atk_names = [f"{scenario.attack}" for scenario in to_plot.keys()]
    atk_names_asr = [f"{atk_names[i]}-{to_plot[scenario]['ASR']:.3f}" for i, scenario in enumerate(to_plot.keys())]

    ax.set_xticklabels([''] + atk_names, rotation=90)
    ax.set_yticklabels([''] + atk_names_asr)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.show()