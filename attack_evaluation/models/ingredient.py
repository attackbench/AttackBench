from functools import partial
from importlib import resources

import torch
from robustbench import load_model
from sacred import Ingredient
from torch import nn

from . import checkpoints
from .mnist import SmallCNN
from .original.utils import load_original_model

model_ingredient = Ingredient('model')


@model_ingredient.config
def config():
    source = 'local'
    requires_grad = False  # if some model requires gradient computations in the forward pass


@model_ingredient.named_config
def mnist_smallcnn():
    name = 'MNIST_SmallCNN'
    dataset = 'mnist'


@model_ingredient.named_config
def mnist_smallcnn_ddn():
    name = 'MNIST_SmallCNN_ddn'
    dataset = 'mnist'


@model_ingredient.named_config
def mnist_smallcnn_trades():
    name = 'MNIST_SmallCNN_trades'
    dataset = 'mnist'


@model_ingredient.named_config
def carmon_2019():
    name = 'Carmon2019Unlabeled'  # 'Carmon2019'
    source = 'robustbench'
    dataset = 'cifar10'
    threat_model = 'Linf'  # training threat model


@model_ingredient.named_config
def augustin_2020():
    name = 'Augustin2020Adversarial'
    source = 'robustbench'
    dataset = 'cifar10'
    threat_model = 'L2'  # training threat model


@model_ingredient.named_config
def standard():
    name = 'Standard'
    source = 'robustbench'
    dataset = 'cifar10'
    threat_model = 'Linf'  # training threat model


@model_ingredient.named_config
def engstrom_2019():
    name = 'Engstrom2019Robustness'
    source = 'robustbench'
    dataset = 'cifar10'
    threat_model = 'L2'  # training threat model. Available [Linf, L2]


@model_ingredient.named_config
def stutz_2020():
    name = 'Stutz2020CCAT'
    source = 'original'
    dataset = 'cifar10'
    threat_model = 'Linf'


@model_ingredient.named_config
def zhang_2020_large():
    name = 'Zhang2020CrownLarge'
    source = 'original'
    dataset = 'cifar10'
    threat_model = 'Linf'


@model_ingredient.named_config
def zhang_2020_small():
    name = 'Zhang2020CrownSmall'
    source = 'original'
    dataset = 'cifar10'
    threat_model = 'Linf'

@model_ingredient.named_config
def xiao_2020():
    name = 'Xiao2020KWTA'
    source = 'original'
    dataset = 'cifar10'
    threat_model = 'Linf'


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
def get_local_model(name: str, dataset: str) -> nn.Module:
    return _local_models[name]()


@model_ingredient.capture
def get_robustbench_model(name: str, dataset: str, threat_model: str) -> nn.Module:
    model = load_model(model_name=name, dataset=dataset, threat_model=threat_model)
    return model


@model_ingredient.capture
def get_original_model(name: str, dataset: str, threat_model: str) -> nn.Module:
    model = load_original_model(model_name=name, dataset=dataset, threat_model=threat_model)
    return model


_model_getters = {
    'local': get_local_model,
    'robustbench': get_robustbench_model,
    'original': get_original_model
}


@model_ingredient.capture
def get_model(source: str, requires_grad: bool = False) -> nn.Module:
    model = _model_getters[source]()
    model.eval()
    model.requires_grad_(requires_grad)
    return model
