from functools import partial
from importlib import resources

import torch
from robustbench import load_model
from sacred import Ingredient
from torch import nn

from . import checkpoints
from .benchmodel_wrapper import BenchModel
from .mnist import SmallCNN
from .original.utils import load_original_model

model_ingredient = Ingredient('model')


@model_ingredient.config
def config():
    source = 'local'
    requires_grad = False  # if some model requires gradient computations in the forward pass
    enforce_box = True  # enforce box constraint by clamping to [0, 1] in model forward
    num_max_propagations = None  # maximum number of forward and backward propagations that the model allows


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
def gowal_2021():
    name = 'Gowal2021Improving_70_16_ddpm_100m'
    source = 'robustbench'
    dataset = 'cifar10'
    threat_model = 'Linf'


@model_ingredient.named_config
def chen_2020():
    name = 'Chen2020Adversarial'
    source = 'robustbench'
    dataset = 'cifar10'
    threat_model = 'Linf'


@model_ingredient.named_config
def debenedetti_2022():
    name = 'Debenedetti2022Light_XCiT-S12'
    source = 'robustbench'
    dataset = 'imagenet'
    threat_model = 'Linf'


@model_ingredient.named_config
def salman_2020():
    name = 'Salman2020Do_50_2'
    source = 'robustbench'
    dataset = 'imagenet'
    threat_model = 'Linf'


@model_ingredient.named_config
def wong_2020():
    name = 'Wong2020Fast'
    source = 'robustbench'
    dataset = 'imagenet'
    threat_model = 'Linf'


@model_ingredient.named_config
def standard_imagenet():
    name = 'Standard_R50'
    source = 'robustbench'
    dataset = 'imagenet'
    threat_model = 'Linf'


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


@model_ingredient.named_config
def wang_2023_small():
    name = 'Wang2023DMAdvSmall'
    source = 'original'
    dataset = 'cifar10'
    threat_model = 'Linf'


@model_ingredient.named_config
def wang_2023_large():
    name = 'Wang2023DMAdvLarge'
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
def get_model(source: str, requires_grad: bool, enforce_box: bool, num_max_propagations: int) -> BenchModel:
    model = BenchModel(_model_getters[source](), enforce_box=enforce_box, num_max_propagations=num_max_propagations)
    model.eval()
    model.requires_grad_(requires_grad)
    return model
