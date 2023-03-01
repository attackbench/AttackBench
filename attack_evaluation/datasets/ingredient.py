from typing import Optional

import numpy as np
from sacred import Ingredient
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST, VisionDataset

dataset_ingredient = Ingredient('dataset')


@dataset_ingredient.config
def config():
    root = 'data'
    num_samples = None  # number of samples to attack, None for all
    random_subset = True  # True for random subset. False for sequential data in Dataset


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


_datasets = {
    'mnist': get_mnist,
    'cifar10': get_cifar10,
}


@dataset_ingredient.capture
def get_dataset(dataset: str):
    return _datasets[dataset]()


@dataset_ingredient.capture
def get_loader(dataset: str, batch_size: int, num_samples: Optional[int] = None,
               random_subset: bool = True) -> DataLoader:
    dataset = get_dataset(dataset=dataset)
    if num_samples is not None:
        if not random_subset:
            dataset = Subset(dataset, indices=list(range(num_samples)))
        else:
            indices = np.random.choice(len(dataset), replace=False, size=num_samples)
            dataset = Subset(dataset, indices=indices)
    loader = DataLoader(dataset=dataset, batch_size=batch_size)
    return loader
