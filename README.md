## Requirements

- python==3.9
- pytorch==1.11.0
- torchvision==0.12.0
- adv_lib https://github.com/jeromerony/adversarial-library
- foolbox
- robustbench https://github.com/RobustBench/robustbench
- sacred

## Download all models

To download all the models and datasets in advance (for instance, to run on a cluster where nodes cannot access internet), run the following command:
```bash
python -m attack_evaluation.run print_named_configs | grep model. | xargs -I{} python -m attack_evaluation.run with dataset.num_samples=0 attack.adv_lib_ddn {}
```

## Usage

To evaluate DDN on a robust (TRADES) SmallCNN model trained on MNIST and save the results in the `results/mnist`
directory:

```bash
python attack_evaluation/print_tables.py -F results/mnist with attack.ddn dataset.mnist model.mnist_smallcnn model.robust=trades
```

## Currently implemented

| Attack  | Original | Advertorch | Adv_lib | ART | CleverHans | DeepRobust | Foolbox | Torchattacks |
|---------|:--------:|:----------:|:-------:|:---:|:----------:|:----------:|:-------:|:------------:|
| DDN     |    ☒     |            |    ✓    |  ☒  |     ☒      |     ☒      |    ✓    |      ☒       |
| ALMA    |    ☒     |     ☒      |    ✓    |  ☒  |     ☒      |     ☒      |    ☒    |      ☒       |
| FMN     |    ✓     |     ☒      |    ✓    |  ☒  |     ☒      |     ☒      |    ✓    |      ☒       |
| PGD     |    ☒     |            |    ✓    |  ✓  |            |     ✓      |         |      ✓       |
| JSMA    |    ☒     |            |    ☒    |  ✓  |     ☒      |     ☒      |    ☒    |      ☒       |
| CW-L2   |    ☒     |            |    ✓    |  ✓  |            |     ~      |    ✓    |      ✓       |
| CW-LINF |    ☒     |     ☒      |    ✓    |  ✓  |     ☒      |     ☒      |    ☒    |      ☒       |
| FGSM    |    ☒     |            |    ☒    |  ✓  |            |            |         |      ✓       |
| BB      |    ☒     |     ☒      |    ☒    |  ✓  |     ☒      |     ☒      |    ✓    |      ☒       |
| DF      |    ✓     |     ☒      |    ☒    |  ✓  |     ☒      |     ~      |    ✓    |      ✓       |
| APGD    |    ✓     |     ☒      |    ✓    |  ✓  |     ☒      |     ☒      |    ☒    |      ✓       |
| BIM     |    ☒     |            |    ☒    |  ✓  |            |     ☒      |         |      ☒       |
| EAD     |    ☒     |            |    ☒    |  ✓  |     ☒      |     ☒      |    ✓    |      ☒       |
| PDGD    |    ☒     |     ☒      |    ✓    |  ☒  |     ☒      |     ☒      |    ☒    |      ☒       |
| PDPGD   |    ☒     |     ☒      |    ✓    |  ☒  |     ☒      |     ☒      |    ☒    |      ☒       |
| TR      |    ✓     |     ☒      |    ✓    |  ☒  |     ☒      |     ☒      |    ☒    |      ☒       |
| FAB     |    ✓     |            |    ✓    |  ☒  |     ☒      |     ☒      |    ☒    |      ✓       |


Legend: 
- _empty_ : not implemented yet 
- ☒ : not available (see google sheet)
- ✓ : implemented
- ~ : not functional yet

## Attack format

To have a standard set of inputs and outputs for all the attacks, the wrappers for all the implementations (including libraries) must have the following format:

- inputs:
    - `model`: `nn.Module` taking inputs in the [0, 1] range and returning logits in $\mathbb{R}^K$
    - `inputs`: `FloatTensor` representing the input samples in the [0, 1] range
    - `labels`: `LongTensor` representing the labels of the samples
    - `targets`: `LongTensor` or `None` representing the targets associated to each samples
    - `targeted`: `bool` flag indicating if a targeted attack should be performed
- output:
    - `adv_inputs`: `FloatTensor` representing the perturbed inputs in the [0, 1] range