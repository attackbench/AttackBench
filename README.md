## Requirements

- python==3.9
- pytorch==1.11.0
- torchvision==0.12.0
- adv_lib https://github.com/jeromerony/adversarial-library
- foolbox
- robustbench https://github.com/RobustBench/robustbench
- sacred

## Usage

To evaluate DDN on a robust (TRADES) SmallCNN model trained on MNIST and save the results in the `results/mnist`
directory:

```bash
python attack_evaluation/run.py -F results/mnist with attack.ddn dataset.mnist model.mnist_smallcnn model.robust=trades
```

## Currently implemented

| Attack | Original | Advertorch | Adv_lib | ART | CleverHans | DeepRobust | Foolbox | Torchattacks |
|--------|:--------:|:----------:|:-------:|:---:|:----------:|:----------:|:-------:|:------------:|
| DDN    |          |            |    ✓    |     |            |            |         |              |
| ALMA   |          |            |    ✓    |     |            |            |         |              |
| FMN    |    ✓     |            |    ✓    |     |            |            |         |              |