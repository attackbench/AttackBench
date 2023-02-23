import argparse
import json
from pathlib import Path

parser = argparse.ArgumentParser(description='Slurm runner for attacks benchmark.')
parser.add_argument('--exp-dir', type=str, default=None, help="Directory where to store results.")
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'mnist'], help='Dataset')
parser.add_argument('--num-samples', type=int, default=None, help='Number of samples for SubDataset.')
parser.add_argument('--model', type=str, default='standard',
                    choices=['standard', 'mnist_smallcnn', 'wideresnet_28_10', 'carmon_2019', 'augustin_2020'],
                    help='Victim model')
parser.add_argument('--threat-model', type=str, default='l2', choices=['l0', 'l1', 'l2', 'linf'], help='Attack norm')
parser.add_argument('--library', type=str, default='all',
                    choices=["foolbox", "art", "adversarial_lib", "torch_attacks", "cleverhans", "deeprobust", "all"],
                    help='Attack library')
parser.add_argument('--account', type=str, default=None, help='Account allocation to use')
parser.add_argument('--device', type=str, default=None,
                    help='Device over which exp are executed. Eg. quadro_rtx_5000, quadro_rtx_5000, tesla')
parser.add_argument('--batch-size', type=int, default=512, help='Batch size')
parser.add_argument('--gpu-count', type=int, default=1, help='Number of gpus for trial')
parser.add_argument('--cpu-count', type=int, default=10, help='Number of cpus for trial')
parser.add_argument('--memory', '--mem', type=int, default=128, help='Number of GB to allocate')
parser.add_argument('--json_attacks', type=str, default='attacks.json', help='JSON file of attacks to run.')
parser.add_argument('--seed', type=int, default=4444, help='Set seed for running experiments.')

args = parser.parse_args()

_conda_env_name = 'atkbench'

if __name__ == "__main__":
    # exp setup
    exp_dir = Path(args.exp_dir or 'results')
    dataset = args.dataset
    num_samples = args.num_samples
    batch_size = args.batch_size
    victim = args.model
    threat_model = args.threat_model
    seed = args.seed

    with open(args.json_attacks, 'r') as f:
        configs = json.load(f)

    for lib in configs[threat_model].keys():

        if lib == args.library or args.library == 'all':

            logs_dir = exp_dir / 'logs' / dataset / threat_model / victim / ('batch_size_' + str(batch_size))
            lib_batch_size = f'{lib}-batch_size_{batch_size}'
            # folder setup
            exp_dir.mkdir(exist_ok=True, parents=True)
            logs_dir.mkdir(exist_ok=True, parents=True)

            library_json = configs[threat_model][lib]
            for attack in library_json['attacks']:
                log_name = f"{lib}-{attack}"
                lines = [
                    "#!/bin/bash",
                    f"#SBATCH --job-name={lib}-{attack}.job",
                    f"#SBATCH --output={Path(logs_dir) / log_name}-log.out",
                    f"#SBATCH --mem={args.memory}G",
                    f"#SBATCH --cpus-per-task={args.cpu_count}",
                ]

                if args.device is not None:
                    sbatch_gpu = f"#SBATCH --gres=gpu:{args.device}:{args.gpu_count}"
                else:
                    sbatch_gpu = f"#SBATCH --gres=gpu:{args.gpu_count}"
                lines.append(sbatch_gpu)

                if args.account is not None:
                    lines.append(f"#SBATCH --account={args.account}")

                lines.append([
                    "# load modules",
                    "module load python/3.9",
                    "source ~/ADV_BENCH/bin/activate",
                    "cd ~/ib_projects/attack_benchmark",
                ])

                attack_name = f"{library_json['prefix']}_{attack}"
                job_file = logs_dir / f'{lib}-runner.job'
                command = f"python attack_evaluation/run.py -F {exp_dir} with " \
                          f"seed={seed} " \
                          f"dataset.{dataset} " \
                          f"dataset.num_samples={num_samples} " \
                          f"dataset.batch_size={batch_size} " \
                          f"model.{victim} " \
                          f"attack.{attack_name} " \
                          f"attack.threat_model={threat_model} "

                with open(job_file, 'w') as fh:
                    fh.write('\n'.join(lines))
