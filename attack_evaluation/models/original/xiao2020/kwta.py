import os
import torch
import urllib.request
from .models import SparseResNet18

MODEL_ID = '1Af_owmMvg1LxjITLE1gFUmPx5idogeTP'


def load_kwta_model(dataset='cifar10', model='kwta_spresnet18', threat_model='Linf'):
    gamma = 0.1
    net_name = f'kwta_spresnet18_{gamma}_cifar_adv.pth'
    model_file = f'models/checkpoints/%s' % net_name
    url = f'https://github.com/wielandbrendel/robustness_workshop/releases/download/v0.0.1/{net_name}'

    if not os.path.exists(model_file):
        print('Downloading pretrained weights.')
        urllib.request.urlretrieve(url, model_file)
    assert os.path.exists(model_file)

    model = SparseResNet18(sparsities=[gamma, gamma, gamma, gamma], sparse_func='vol')
    state_dict = torch.load(model_file, map_location='cpu')
    model.load_state_dict(state_dict)
    return model
