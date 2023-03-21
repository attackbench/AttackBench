import torch
from .wideresnetwithswish import wideresnetwithswish
import os
import urllib.request

_pretrained_model_info = {'cifar10': {
    'Wang2023DMAdvSmall': {
        'name': 'wrn-28-10-swish',
        'constructor': wideresnetwithswish,
        'checkpoint_url': dict(
            Linf='https://huggingface.co/wzekai99/DM-Improves-AT/resolve/main/checkpoint/cifar10_linf_wrn28-10.pt',
            L2='https://huggingface.co/wzekai99/DM-Improves-AT/resolve/main/checkpoint/cifar10_l2_wrn28-10.pt')},

    'Wang2023DMAdvLarge': {
        'name': 'wrn-70-16-swish',
        'constructor': wideresnetwithswish,
        'checkpoint_url': dict(
            Linf='https://huggingface.co/wzekai99/DM-Improves-AT/resolve/main/checkpoint/cifar10_linf_wrn70-16.pt',
            L2='https://huggingface.co/wzekai99/DM-Improves-AT/resolve/main/checkpoint/cifar10_l2_wrn70-16.pt')
    }}}


def download_model(dataset='cifar10', model='stutz_2020', threat_model='LInf'):
    model_info = _pretrained_model_info[dataset][model]

    url = model_info['checkpoint_url'][threat_model]
    model_file = f'models/checkpoints/%s' % url.split('/')[-1]

    urllib.request.urlretrieve(url, model_file)
    assert os.path.exists(model_file)


def create_model(checkpoint_path, model_info):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model = model_info['constructor'](model_info['name'])

    model = torch.nn.Sequential(model)
    model = torch.nn.DataParallel(model)
    # checkpoints are stored in dataparallel modules.
    model.load_state_dict(checkpoint['model_state_dict'])
    return model.module


def load_dm_adv_model(dataset='cifar10', model='stutz2020', threat_model='Linf'):
    assert dataset in _pretrained_model_info.keys()
    assert model in _pretrained_model_info[dataset].keys()
    assert threat_model in _pretrained_model_info[dataset][model]['checkpoint_url'].keys()

    model_info = _pretrained_model_info[dataset][model]

    url = model_info['checkpoint_url'][threat_model]
    model_file = f'models/checkpoints/%s' % url.split('/')[-1]

    if not os.path.exists(model_file):
        print('Downloading pretrained weights.')
        download_model(dataset, model, threat_model)

    model = create_model(model_file, model_info)
    return model
