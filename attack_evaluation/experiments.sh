#!/bin/bash

attacks='fmn ddn alma'
dist_name='norm'

for attack in $attacks
do
  for n in 0 1 2 'inf'
  do
    if [[ $attack == 'alma' ]]
    then
      dist_name='distance'
      n="l$n"
    fi

    python run.py -F "mnist-mnist_smallcnn-$attack-l$n" with save_adv dataset.mnist model.mnist_smallcnn attack.$attack attack.$dist_name=$n
  done
  echo ''
done
