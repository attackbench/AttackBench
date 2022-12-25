#!/bin/bash

#python job_runner.py --model='carmon_2019' # cifar10
#python job_runner.py --model='wideresnet_28_10' # cifar10
#python job_runner.py --dataset="mnist" --model='mnist_smallcnn' # mnist

# Test attacks with their default parameters on a subset of the test set (1000).
# Note: num_steps is the only parameter changed to have fair comparison between different implementation of the
# same attack algorithm.
python job_runner.py --exp_dir=results --threat_model=l2 --model=standard --batch_size=128 --num_samples=1000 --seed=4444
