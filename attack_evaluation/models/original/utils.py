from .stutz2020.ccat import load_ccat_model
from .zhang2020.crown import load_crown_model

_available_defenses = {'Stutz2020CCAT': load_ccat_model,
                       'Zhang2020CrownLarge': load_crown_model,
                       'Zhang2020CrownSmall': load_crown_model}


def load_original_model(model_name: str, dataset: str, threat_model: str):
    return _available_defenses[model_name](dataset, model=model_name, threat_model=threat_model)
