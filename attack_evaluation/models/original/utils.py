from .stutz2020.ccat import load_model
from functools import partial

_available_defenses = {'Stutz2020CCAT': partial(load_model)}


def load_original_model(model_name: str, dataset: str, threat_model: str):
    return _available_defenses[model_name](dataset, model=model_name, threat_model=threat_model)
