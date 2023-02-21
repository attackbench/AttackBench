from functools import partial
from importlib import resources

import torch
from robustbench import load_model
from sacred import Ingredient
from torch import nn

from . import checkpoints
from .mnist import SmallCNN

model_ingredient = Ingredient('model')


@model_ingredient.config
def config():
    source = 'local'
    requires_grad = False  # if some model requires gradient computations in the forward pass


@model_ingredient.named_config
def mnist_smallcnn():
    name = 'MNIST_SmallCNN'


@model_ingredient.named_config
def mnist_smallcnn_ddn():
    name = 'MNIST_SmallCNN_ddn'


@model_ingredient.named_config
def mnist_smallcnn_trades():
    name = 'MNIST_SmallCNN_trades'


@model_ingredient.named_config
def carmon_2019():
    name = 'Carmon2019Unlabeled'  # 'Carmon2019'
    source = 'robustbench'


@model_ingredient.named_config
def augustin_2020():
    name = 'Augustin2020'
    source = 'robustbench'


@model_ingredient.named_config
def standard():
    name = 'Standard'
    source = 'robustbench'


@model_ingredient.capture
def get_mnist_smallcnn(checkpoint: str) -> nn.Module:
    model = SmallCNN()
    with resources.path(checkpoints, checkpoint) as f:
        state_dict = torch.load(f, map_location='cpu')
    model.load_state_dict(state_dict)
    return model


_local_models = {
    'MNIST_SmallCNN': partial(get_mnist_smallcnn, checkpoint='mnist_smallcnn_standard.pth'),
    'MNIST_SmallCNN_ddn': partial(get_mnist_smallcnn, checkpoint='mnist_smallcnn_robust_ddn.pth'),
    'MNIST_SmallCNN_trades': partial(get_mnist_smallcnn, checkpoint='mnist_smallcnn_robust_trades.pth'),
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
def get_model(source: str, requires_grad: bool = False) -> nn.Module:
    model = _model_getters[source]()
    model.eval()

    for param in model.parameters():
        param.requires_grad_(requires_grad)

    return model
