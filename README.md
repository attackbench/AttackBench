# **AttackBench**: Evaluating Gradient-based Attacks for Adversarial Examples

Antonio Emanuele Cinà $^\star$, Jérôme Rony $^\star$, Maura Pintor, Luca Demetrio, Ambra Demontis, Battista Biggio, Ismail Ben Ayed, and Fabio Roli 

**Leaderboard**: [https://attackbench.github.io/](https://attackbench.github.io/)

**Paper:** [https://arxiv.org/pdf/2404.19460](https://arxiv.org/pdf/2404.19460)

## How it works

The <code>AttackBench</code> framework wants to fairly compare gradient-based attacks based on their security evaluation curves. To this end, we derive a process involving five distinct stages, as depicted below.
  - In stage (1), we construct a list of diverse non-robust and robust models to assess the attacks' impact on various settings, thus testing their adaptability to diverse defensive strategies. 
  - In stage (2), we define an environment for testing gradient-based attacks under a systematic and reproducible protocol. 
        This step provides common ground with shared assumptions, advantages, and limitations. 
        We then run the attacks against the selected models individually and collect the performance metrics of interest in our analysis, which are perturbation size, execution time, and query usage. 
  - In stage (3), we gather all the previously-obtained results, comparing  attacks with the novel <code>local optimality</code> metric. 
  - Finally, in stage (4), we aggregate the optimality results from all considered models, and in stage (5) we rank the attacks based on their average optimality, namely <code>global optimality</code>. 
  

<p align="center"><img src="https://attackbench.github.io/assets/AtkBench.svg" width="1300"></p>


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
- ☒ : not available
- ✓ : implemented
- ~ : not functional yet



## Requirements and Installations 

- python==3.9
- sacred
- pytorch==1.12.1
- torchvision==0.13.1
- adversarial-robustness-toolbox
- foolbox
- torchattacks
- cleverhans
- deeprobust
- robustbench https://github.com/RobustBench/robustbench
- adv_lib https://github.com/jeromerony/adversarial-library

Clone the Repository:
```bash
git clone https://github.com/attackbench/attackbench.git
cd attackbench
```

Use the provided `environment.yml` file to create a Conda environment with the required dependencies:
```bash
conda env create -f environment.yml
```

Activate the Conda environment: 
```bash
conda activate attackbench
```


## Usage

To run the FMN-$\ell_2$ attack implemented within the <code>adversarial lib</code> library against the <code>augustin_2020</code> DDN on CIFAR10 and save the results in the `results_dir/` directory:

```bash
conda activate attackbench
python -m attack_evaluation.run  -F results_dir/ with model.augustin_2020 attack.adv_lib_fmn attack.threat_model="l2" dataset.num_samples=1000 dataset.batch_size=64 seed=42
```

Command Breakdown:
- `-F results_dir/`: Specifies the directory results_dir/ where the attack results will be saved.
- `with`: Keyword for sacred.
- `model.augustin_2020`: Specifies the target model augustin_2020 to be attacked.
- `attack.adv_lib_fmn`: Indicates the use of the FMN attack from the adv_lib library.
- `attack.threat_model="l2"`: Sets the threat model to $\ell_2$, constraining adversarial perturbations based on the $\ell_2$ norm.
- `dataset.num_samples=1000`: Specifies the number of samples to use from the CIFAR-10 dataset during the attack.
- `dataset.batch_size=64`: Sets the batch size for processing the dataset during the attack.
- `seed=42`: Sets the random seed for reproducibility.

After the attack completes, you can find the results saved in the specified results_dir/ directory.



## Attack format

Tthe wrappers for all the implementations (including libraries) must have the following format:

- inputs:
    - `model`: `nn.Module` taking inputs in the [0, 1] range and returning logits in $\mathbb{R}^K$
    - `inputs`: `FloatTensor` representing the input samples in the [0, 1] range
    - `labels`: `LongTensor` representing the labels of the samples
    - `targets`: `LongTensor` or `None` representing the targets associated to each samples
    - `targeted`: `bool` flag indicating if a targeted attack should be performed
- output:
    - `adv_inputs`: `FloatTensor` representing the perturbed inputs in the [0, 1] range


## Citation

If you use the **AttackBench** leaderboards or implementation, then consider citing our [paper]():

```bibtex
@inproceedings{cina2024attackbench,
	author = {Cin{\`a}, A. E. and Rony, J. and Pintor, M. and Demetrio, L. and Demontis, A. and Biggio, B. and Ayed, I. B. and Roli, F.},
  title = {AttackBench: Evaluating Gradient-based Attacks for Adversarial Examples},
  booktitle={AAAI Conference on Artificial Intelligence},
  year = {2025},
}
```

## Contact 
Feel free to contact us about anything related to **`AttackBench`** by creating an issue, a pull request or 
by email at `antonio.cina@unige.it`.

## Acknowledgements
AttackBench has been partially developed with the support of European Union’s [ELSA – European Lighthouse on Secure and Safe AI](https://elsa-ai.eu), Horizon Europe, grant agreement No. 101070617, and [Sec4AI4Sec - Cybersecurity for AI-Augmented Systems](https://www.sec4ai4sec-project.eu), Horizon Europe, grant agreement No. 101120393.

<img src="_static/assets/logos/sec4AI4sec.png" alt="sec4ai4sec" style="width:70px;"/> &nbsp;&nbsp; 
<img src="_static/assets/logos/elsa.jpg" alt="elsa" style="width:70px;"/> &nbsp;&nbsp; 
<img src="_static/assets/logos/FundedbytheEU.png" alt="europe" style="width:240px;"/>
