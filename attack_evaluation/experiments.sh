#!/bin/bash

python job_runner.py --model='carmon_2019' # cifar10
python job_runner.py --model='wideresnet_28_10' # cifar10

python job_runner.py --dataset="mnist" --model='mnist_smallcnn' # mnist