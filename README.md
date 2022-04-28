To evaluate DDN on a robust (TRADES) SmallCNN model trained on MNIST and save the results in the `results/mnist` directory:
```bash
❯ python attack_evaluation/experiment.py -F results/mnist with attack.ddn dataset.mnist model.mnist_smallcnn model.robust=trades
```