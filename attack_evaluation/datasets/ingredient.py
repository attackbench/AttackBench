from typing import Optional

from sacred import Ingredient
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, MNIST, VisionDataset

dataset_ingredient = Ingredient('dataset')


@dataset_ingredient.config
def config():
    root = 'data'
    num_samples = None  # number of samples to attack, None for all


@dataset_ingredient.named_config
def mnist():
    name = 'mnist'
    batch_size = 10000


@dataset_ingredient.named_config
def cifar10():
    name = 'cifar10'
    batch_size = 256


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
def get_dataset(name: str):
    return _datasets[name]()


@dataset_ingredient.capture
def get_loader(batch_size: int, num_samples: Optional[int] = None) -> DataLoader:
    dataset = get_dataset()
    if num_samples is not None:
        dataset = Subset(dataset, indices=list(range(num_samples)))
    loader = DataLoader(dataset=dataset, batch_size=batch_size)
    return loader
