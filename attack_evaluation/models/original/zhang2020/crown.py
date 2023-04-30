import os
import tarfile

import requests
from adv_lib.utils import normalize_model
from torch import nn

from .model_defs_gowal import IBP_large, model_cnn_4layer
from .utils import load_crown_dict

_pretrained_model_configs = {
    'Zhang2020CrownLarge': dict(prefix='_dm-large',
                                model_path="cifar_dm-large_8_255/IBP_large_best.pth",
                                model_class="IBP_large",
                                model_params={"in_ch": 3, "in_dim": 32, "linear_size": 512}),
    'Zhang2020CrownSmall': dict(prefix="",
                                model_path="cifar_crown_0.03137/cifar_8_small/cnn_4layer_linear_512_width_1_best.pth",
                                model_class="model_cnn_4layer",
                                model_params={"in_ch": 3, "in_dim": 32, "width": 1, "linear_size": 512})
}


def download_model(model: str, dataset: str = 'cifar10') -> None:
    pretrained_config = _pretrained_model_configs[model]

    url = f"https://download.huan-zhang.com/models/crown-ibp/models_crown-ibp{pretrained_config['prefix']}.tar.gz"
    response = requests.get(url, stream=True)
    file = tarfile.open(fileobj=response.raw, mode="r|gz")
    file.extractall(path="./models/checkpoints/")


def load_crown_model(model: str, dataset: str = 'cifar10', threat_model: str = 'Linf') -> nn.Module:
    assert dataset in ['cifar10']  # available ['mnist']

    config = _pretrained_model_configs[model]
    if model == 'Zhang2020CrownLarge':
        net = IBP_large(**config['model_params'])
        model_file = "./models/checkpoints/models_crown-ibp_dm-large/"
    else:
        net = model_cnn_4layer(**config['model_params'])
        model_file = "./models/checkpoints/crown-ibp_models/"

    if not os.path.exists(model_file):
        download_model(model=model, dataset=dataset)
    assert os.path.exists(model_file)

    model_dict = load_crown_dict(model_file + config['model_path'])
    net.load_state_dict(model_dict)
    return normalize_model(net, mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010))
