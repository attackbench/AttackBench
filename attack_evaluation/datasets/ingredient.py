from typing import Optional
from pathlib import Path

import numpy as np
from sacred import Ingredient
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST, VisionDataset
from .utils import ImageNetKaggle

dataset_ingredient = Ingredient('dataset')


@dataset_ingredient.config
def config():
    root = 'data'
    num_samples = None  # number of samples to attack, None for all
    random_subset = True  # True for random subset. False for sequential data in Dataset
    batch_size = 128


@dataset_ingredient.capture
def get_mnist(root: str) -> VisionDataset:
    transform = transforms.ToTensor()
    dataset = MNIST(root=root, train=False, transform=transform, download=True)
    return dataset


@dataset_ingredient.capture
def get_cifar10(root: str) -> VisionDataset:
    transform = transforms.ToTensor()
    dataset = CIFAR10(root=root, train=False, transform=transform, download=True)
    return dataset

@dataset_ingredient.capture
def get_imagenet(root: str) -> VisionDataset:
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])
    data_path = Path(root) / 'imagenet-data'# / 'val'
    dataset = ImageNetKaggle(root=data_path, split='val', transform=transform)#ImageNet(root=data_path, transform=transform)
    return dataset


_datasets = {
    'mnist': get_mnist,
    'cifar10': get_cifar10,
    'imagenet': get_imagenet,
}


@dataset_ingredient.capture
def get_dataset(dataset: str):
    return _datasets[dataset]()


@dataset_ingredient.capture
def get_loader(dataset: str, batch_size: int, num_samples: Optional[int] = None,
               random_subset: bool = True) -> DataLoader:
    data = get_dataset(dataset=dataset)

    if num_samples is not None:
        if not random_subset:
            data = Subset(data, indices=list(range(num_samples)))
        else:
            indices = np.random.choice(len(data), replace=False, size=num_samples)
            data = Subset(data, indices=indices)
    loader = DataLoader(dataset=data, batch_size=batch_size)
    return loader

