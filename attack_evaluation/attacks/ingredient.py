import inspect
import sys
from collections import defaultdict
from functools import partial
from typing import Callable

from sacred import Ingredient

from .adv_lib import configs as adv_lib_configs
from .art import configs as art_configs
from .cleverhans import configs as cleverhans_configs
from .deeprobust import configs as deeprobust_configs
from .foolbox import configs as foolbox_configs
from .original import configs as original_configs
from .torchattacks import configs as torchattacks_configs

attack_ingredient = Ingredient('attack')

library_modules = {
    'adv_lib': adv_lib_configs,
    'art': art_configs,
    'cleverhans': cleverhans_configs,
    'deeprobust': deeprobust_configs,
    'foolbox': foolbox_configs,
    'original': original_configs,
    'torchattacks': torchattacks_configs,
}
library_getters = defaultdict(dict)

for module_name, module in library_modules.items():
    # gather function defined in <library>.configs modules
    module_funcs = inspect.getmembers(sys.modules[module.__name__],
                                      predicate=lambda f: inspect.isfunction(f) and f.__module__ == module.__name__)

    for name, func in module_funcs:  # search for functions that are configs or getters
        config_prefix = module._prefix + '_'
        getter_prefix = 'get_' + config_prefix

        if name.startswith(config_prefix):  # add function as named config
            attack_ingredient.named_config(func)

        elif name.startswith(getter_prefix):  # capture function and add it to the getters
            library_getters[module_name][name.removeprefix(getter_prefix)] = attack_ingredient.capture(func)


@attack_ingredient.capture
def get_attack(source: str, name: str, threat_model: str) -> Callable:
    attack = library_getters[source][name]()
    wrapper = library_modules[source]._wrapper

    if isinstance(attack, dict):
        return partial(wrapper, **attack)
    else:
        return partial(wrapper, attack=attack)
