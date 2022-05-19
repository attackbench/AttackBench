#!/bin/bash

#SBATCH --job-name=cifar10_l2   # Job name
#SBATCH --ntasks=40                    # Run on a single CPU
#SBATCH --mem=128gb                     # Job memory request
#SBATCH --time=48:00:00               # Time limit hrs:min:sec
#SBATCH --output=cifar10_l2.log   # Standard output and error log
#SBATCH --gres gpu:quadro_rtx_8000:1

python run.py -F cifar-standard-fmn-l2 with save_adv dataset.cifar10 model.standard model.threat_model='L2' attack.fmn attack.norm=2
python run.py -F cifar-standard-alma-l2 with save_adv dataset.cifar10 model.standard model.threat_model='L2' attack.alma attack.norm=2
python run.py -F cifar-standard-ddn-l2 with save_adv dataset.cifar10 model.standard model.threat_model='L2' attack.ddn attack.norm=2

python run.py -F cifar-standard-fmn-l0 with save_adv dataset.cifar10 model.standard model.threat_model='L2' attack.fmn attack.norm=0
python run.py -F cifar-standard-ddn-l0 with save_adv dataset.cifar10 model.standard model.threat_model='L2' attack.ddn attack.norm=0

python run.py -F cifar-standard-fmn-l1 with save_adv dataset.cifar10 model.standard model.threat_model='L2' attack.fmn attack.norm=1
python run.py -F cifar-standard-alma-l1 with save_adv dataset.cifar10 model.standard model.threat_model='L2' attack.alma attack.norm=1
python run.py -F cifar-standard-ddn-l1 with save_adv dataset.cifar10 model.standard model.threat_model='L2' attack.ddn attack.norm=1

python run.py -F cifar-standard-fmn-linf with save_adv dataset.cifar10 model.standard model.threat_model='L2' attack.fmn attack.norm=inf
python run.py -F cifar-standard-ddn-linf with save_adv dataset.cifar10 model.standard model.threat_model='L2' attack.ddn attack.norm=inf
