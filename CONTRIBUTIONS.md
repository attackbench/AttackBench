
- write a user guide where we specify how to integrate new attacks, and how use the benchmark

- analysis.py --> fetch the exp name for config.json (it contains all configs from the exp)

- analysis.py --> use info.json to get the metadata about the attacks (it does not contain adv examples)

- analysis.py --> implement the empirical optimal attack curve

- experiments: export metadata for each sample (norm, success, etc...)

- remove foolbox wrapper

- attacks:
    - Voting Folded Gaussian Attack (VFGA)
    - Primal Dual Proximal Gradient Descent (PDPGD) L0 
    - baseline for empirical optimal attack

------------------------------------------------

- remove seaborn 

- use less libraries as possible 
