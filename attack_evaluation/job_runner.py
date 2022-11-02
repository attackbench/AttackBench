from pathlib import Path
import argparse
import os
import json

parser = argparse.ArgumentParser(description='Slurm runner for attacks benchmark.')
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'mnist'], help='Dataset')
parser.add_argument('--model', type=str, default='standard',
                    choices=['standard', 'mnist_smallcnn', 'wideresnet_28_10', 'carmon_2019', 'augustin_2020'],
                    help='Victim model')
parser.add_argument('--threat_model', type=str, default='l2', choices=['l0', 'l1', 'l2', 'linf'], help='Attack norm')
parser.add_argument('--library', type=str, default='all',
                    choices=["foolbox", "art", "adversarial_lib", "torch_attacks", "cleverhans", "deeprobust", "all"], help='Attack library')
parser.add_argument('--device', type=str, default='quadro_rtx_5000',
                    help='Device over which exp are executed. Eg. quadro_rtx_5000, quadro_rtx_5000, tesla')
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
    root = Path('results')
    dataset = args.dataset
    batch_size = args.batch_size
    victim = args.model

    with open('attacks.json', 'r') as f:
        configs = json.load(f)

    for lib in configs[args.threat_model].keys():

        if lib == args.library or args.library == 'all':

            exp_dir = root  # / dataset / args.threat_model / victim
            logs_dir = root / 'logs' / dataset
            lib_batch_size = f'{lib}-batch_size_{batch_size}'
            # folder setup
            root.mkdir(exist_ok=True)
            exp_dir.mkdir(exist_ok=True, parents=True)
            logs_dir.mkdir(exist_ok=True, parents=True)

            library_json = configs[args.threat_model][lib]
            for attack in library_json['attacks']:
                attack_name = f"{library_json['prefix']}_{attack}"

                log_name = f"{lib}-{attack}"

                job_file = exp_dir / f'{lib}-runner.job'
                command = f"python run.py -F {exp_dir / lib_batch_size} with " \
                          f"dataset.{dataset} " \
                          f"dataset.batch_size={batch_size} " \
                          f"model.{victim} " \
                          f"attack.{attack_name} " \
                          f"attack.threat_model={args.threat_model} "
                lines = [
                    "#!/bin/bash",
                    f"#SBATCH --job-name={lib}-{attack}.job",
                    f"#SBATCH --output={Path(logs_dir) / log_name}-log.out",
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
