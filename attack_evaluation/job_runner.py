import argparse
import os
from pathlib import Path

parser = argparse.ArgumentParser(description='Slurm runner for attacks benchmark.')
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'mnist'], help='Dataset')
parser.add_argument('--attacks', type=list, default=['adv_lib_fmn'], help='Evasion attack')
parser.add_argument('--model', type=str, default='standard',
                    choices=['standard', 'mnist_smallcnn', 'wideresnet_28_10', 'carmon_2019', 'augustin_2020'],
                    help='Victim model')
parser.add_argument('--norm', type=str, default='l2', choices=['l0', 'l1', 'l2', 'linf'], help='Attack norm')
parser.add_argument('--device', type=str, default='quadro_rtx_8000',
                    help='Device over which exp are executed. Eg. quadro_rtx_8000, quadro_rtx_5000, tesla')
parser.add_argument('--batch_size', type=int, default=512, help='Batch size')
parser.add_argument('--gpu_count', type=int, default=1, help='Number of gpus for trial')
parser.add_argument('--cpu_count', type=int, default=10, help='Number of cpus for trial')
parser.add_argument('--memory', '--mem', type=int, default=128, help='Number of GB to allocate')
args = parser.parse_args()

_conda_env_name = 'atkbench'

if __name__ == "__main__":
    # machine setup
    device = args.device
    gpu_count = args.gpu_count
    cpu_count = args.cpu_count
    memory = args.memory

    # exp setup
    root = Path('experimental_results')
    dataset = args.dataset
    batch_size = args.batch_size
    attacks = args.attacks
    victim = args.model
    norm = float(args.norm[1:])
    norm_str = args.norm

    for attack in attacks:
        exp_dir = root / dataset / norm_str / victim
        logs_dir = root / 'logs' / dataset / norm_str / victim

        # folder setup
        root.mkdir(exist_ok=True)
        exp_dir.mkdir(exist_ok=True, parents=True)
        logs_dir.mkdir(exist_ok=True, parents=True)

        job_file = exp_dir / f'{attack}-runner.job'
        command = f"python run.py -F {exp_dir / attack} with " \
                  f"dataset.{dataset} " \
                  f"dataset.batch_size={batch_size} " \
                  f"model.{victim} " \
                  f"attack.{attack} " \
                  f"attack.norm={norm}"
        # f"model.threat_model=L{norm[1:]} "\
        lines = [
            "#!/bin/bash",
            f"#SBATCH --job-name={dataset}-{attack}.job",
            f"#SBATCH --output={Path(logs_dir) / attack}-log.out",
            f"#SBATCH --mem={memory}gb",
            f"#SBATCH --ntasks={cpu_count}",
            f"#SBATCH --gres gpu:{device}:{gpu_count}",
            command,
        ]

        with open(job_file, 'w') as fh:
            fh.write('\n'.join(lines))

        print(f'Running {command} ...')
        os.system("sbatch %s" % job_file)
        print(f'Job Started')
