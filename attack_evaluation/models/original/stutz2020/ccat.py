"""
Code adapted from: https://github.com/davidstutz/confidence-calibrated-adversarial-training
"""

import wget
import zipfile
import os
import torch
from .resnet import ResNet

_pretrained_model_names = {'Stutz2020CCAT': 'ccat'}

_model_name = 'stutz_2020_ccat.pth.tar'


def download_model(dataset='cifar10', model='stutz_2020'):
    model_name = _pretrained_model_names[model]

    url = 'https://datasets.d2.mpi-inf.mpg.de/arxiv2019-ccat/%s_%s.zip' % (dataset, model_name)
    filename = wget.download(url)

    # Directory to extract the model to.
    model_dir = './models/checkpoints/'
    with zipfile.ZipFile(filename, 'r') as zip_ref:
        zip_ref.extractall(model_dir)
    os.remove(filename)
    os.rename(model_dir + zip_ref.infolist()[0].filename, model_dir + _model_name)


def load_ccat_model(dataset='cifar10', model='stutz2020', threat_model='Linf'):
    assert dataset in ['cifar10']  # , 'svhn', 'mnist']
    # assert model in ['normal', 'at', 'stutz_2020', 'msd']

    model_file = 'models/checkpoints/%s' % _model_name
    if not os.path.exists(model_file):
        download_model(dataset, model)
    assert os.path.exists(model_file)

    checkpoint = torch.load(model_file, map_location=lambda storage, loc: storage)
    arguments = {**checkpoint['arguments'], **checkpoint['kwargs']}
    model = ResNet(**arguments)
    model.load_state_dict(checkpoint['model'])
    return model

