#!/bin/bash

#python job_runner.py --model='carmon_2019' # cifar10
#python job_runner.py --model='wideresnet_28_10' # cifar10
#python job_runner.py --dataset="mnist" --model='mnist_smallcnn' # mnist

# Test attacks with their default parameters on a subset of the test set (1000) on standard model.
# Note: num_steps is the only parameter changed to have fair comparison between different implementation of the
# same attack algorithm.
python job_runner.py --exp_dir=results5000 --threat_model=l2 --model=standard --batch_size=128 --num_samples=1000 --seed=4444
python job_runner.py --exp_dir=results5000 --threat_model=linf --model=standard --batch_size=128 --num_samples=1000 --seed=4444
python job_runner.py --exp_dir=results1000 --threat_model=l0 --model=standard --batch_size=128 --num_samples=1000 --seed=4444
python job_runner.py --exp_dir=results1000 --threat_model=l1 --model=standard --batch_size=128 --num_samples=1000 --seed=4444

python job_runner.py --exp_dir=results5000 --threat_model=l2 --model=standard --batch_size=1 --num_samples=1000 --seed=4444
python job_runner.py --exp_dir=results5000 --threat_model=linf --model=standard --batch_size=1 --num_samples=1000 --seed=4444
python job_runner.py --exp_dir=results1000 --threat_model=l0 --model=standard --batch_size=1 --num_samples=1000 --seed=4444
python job_runner.py --exp_dir=results1000 --threat_model=l1 --model=standard --batch_size=1 --num_samples=1000 --seed=4444

# Test attacks with their default parameters on a subset of the test set (1000) on carmon model.
# Note: num_steps is the only parameter changed to have fair comparison between different implementation of the
# same attack algorithm.
python job_runner.py --exp_dir=results5000 --threat_model=l2 --model=carmon_2019 --batch_size=128 --num_samples=1000 --seed=4444
python job_runner.py --exp_dir=results5000 --threat_model=linf --model=carmon_2019 --batch_size=128 --num_samples=1000 --seed=4444
python job_runner.py --exp_dir=results1000 --threat_model=l0 --model=carmon_2019 --batch_size=128 --num_samples=1000 --seed=4444
python job_runner.py --exp_dir=results1000 --threat_model=l1 --model=carmon_2019 --batch_size=128 --num_samples=1000 --seed=4444

python job_runner.py --exp_dir=results5000 --threat_model=l2 --model=carmon_2019 --batch_size=1 --num_samples=1000 --seed=4444
python job_runner.py --exp_dir=results5000 --threat_model=linf --model=carmon_2019 --batch_size=1 --num_samples=1000 --seed=4444
python job_runner.py --exp_dir=results1000 --threat_model=l0 --model=carmon_2019 --batch_size=1 --num_samples=1000 --seed=4444
python job_runner.py --exp_dir=results1000 --threat_model=l1 --model=carmon_2019 --batch_size=1 --num_samples=1000 --seed=4444