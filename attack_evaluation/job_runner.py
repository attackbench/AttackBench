import os
from pathlib import Path
import argparse
from utils import mkdir_p

parser = argparse.ArgumentParser(description='Slurm runner for attacks benchmark.')
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'mnist'], help='Dataset')
parser.add_argument('--attacks', type=list, default=['fmn', 'alma', 'ddn'], help='Evasion attack')
parser.add_argument('--model', type=str, default='standard',
                    choices=['standard', 'mnist_smallcnn', 'wideresnet_28_10', 'carmon_2019', 'augustin_2020'],
                    help='Victim model')
parser.add_argument('--norm', type=str, default='l2', choices=['l0', 'l1', 'l2', 'linf'], help='Attack norm')
parser.add_argument('--device', type=str, default='quadro_rtx_8000',
                    help='Device over which exp are executed. Eg. quadro_rtx_8000, quadro_rtx_5000, tesla')
parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
parser.add_argument('--gpu_count', type=int, default=1, help='Number of gpus for trial')
parser.add_argument('--cpu_count', type=int, default=10, help='Number of cpus for trial')
args = parser.parse_args()

_conda_env_name = 'atkbench'


_attacks_keywords = {'fmn': {'distance': 'norm', 'encode': lambda x: x[1:]},
                     'alma': {'distance': 'distance', 'encode': lambda x: x},
                     'ddn': {'distance': 'norm', 'encode': lambda x: x[1:]}
                     }

if __name__ == "__main__":
    # machine setup
    device = args.device
    gpu_count = args.gpu_count
    cpu_count = args.cpu_count

    # exp setup
    root = 'experimental_results'
    dataset = args.dataset
    batch_size = args.batch_size
    attacks = args.attacks
    victim = args.model
    norm = args.norm

    for attack in attacks:
        attack_name = f'{attack}-{norm}'
        exp_dir = f'{root}/{dataset}'
        exp_name = f'{dataset}-{victim}-{attack_name}'
        logs_dir = Path(exp_dir) / 'logs'

        # folder setup
        mkdir_p(root)
        mkdir_p(exp_dir)
        mkdir_p(logs_dir)

        job_file = Path(exp_dir) / Path(f'{attack_name}-runner.job')
        Path(logs_dir) / attack_name
        with open(job_file, 'w') as fh:
            fh.writelines("#!/bin/bash\n")
            fh.writelines(f"#SBATCH --job-name={dataset}-{attack}.job\n")
            fh.writelines(f"#SBATCH --output={Path(logs_dir) / attack_name}-log.out\n")
            fh.writelines("#SBATCH --mem=128gb\n")
            fh.writelines("#SBATCH --ntasks=%d\n" % cpu_count)
            fh.writelines(f"#SBATCH --gres gpu:{device}:{gpu_count}\n")
            fh.writelines(f"python run.py -F {Path(exp_dir)/exp_name} with "
                          f"save_adv "
                          f"dataset.{dataset} "
                          f"dataset.batch_size={batch_size} "
                          f"model.{victim} "
                          f"model.threat_model=L{norm[1:]} "
                          f"attack.{attack} "
                          f"attack.steps=1 "
                          f"attack.{_attacks_keywords[attack]['distance']}={_attacks_keywords[attack]['encode'](norm)}\n")

        os.system("sbatch %s" % job_file)
