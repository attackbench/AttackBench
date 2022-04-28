from importlib import resources
from typing import Optional

import torch
from robustbench import load_model
from sacred import Ingredient
from torch import nn

from . import checkpoints
from .mnist import SmallCNN

model_ingredient = Ingredient('model')


@model_ingredient.named_config
def config():
    requires_grad = False  # if some model requires gradient computations in the forward pass


@model_ingredient.named_config
def mnist_smallcnn():
    name = 'MNIST_SmallCNN'
    origin = 'local'
    robust = None


@model_ingredient.named_config
def wideresnet_28_10():
    name = 'Standard'
    origin = 'robustbench'


@model_ingredient.named_config
def carmon_2019():
    name = 'Carmon2019'
    origin = 'robustbench'


@model_ingredient.named_config
def augustin_2020():
    name = 'Augustin2020'
    origin = 'robustbench'


_mnist_checkpoints = {
    None: 'mnist_smallcnn_standard.pth',
    'ddn': 'mnist_smallcnn_robust_ddn.pth',
    'trades': 'mnist_smallcnn_robust_trades.pth',
}


@model_ingredient.capture
def get_mnist_smallcnn(robust: Optional[str] = None) -> nn.Module:
    model = SmallCNN()
    checkpoint_file = _mnist_checkpoints[robust]
    with resources.path(checkpoints, checkpoint_file) as f:
        state_dict = torch.load(f, map_location='cpu')
    model.load_state_dict(state_dict)
    return model


_local_models = {
    'MNIST_SmallCNN': get_mnist_smallcnn,
}


@model_ingredient.capture
def get_local_model(name: str) -> nn.Module:
    return _local_models[name]()


@model_ingredient.capture
def get_robustbench_model(name: str) -> nn.Module:
    model = load_model(model_name=name)
    return model


_model_getters = {
    'local': get_local_model,
    'robustbench': get_robustbench_model,
}


@model_ingredient.capture
def get_model(origin: str, requires_grad: bool = False) -> nn.Module:
    model = _model_getters[origin]()
    model.eval()

    for param in model.parameters():
        param.requires_grad_(requires_grad)

    return model
