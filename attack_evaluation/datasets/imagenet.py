import json
import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, Subset

from . import subsets


class ImageNetKaggle(Dataset):
    def __init__(self, root, split='val', transform=None):
        self.samples = []
        self.targets = []
        self.transform = transform
        self.syn_to_class = {}
        self.split = split

        with open(os.path.join(root, "imagenet_class_index.json"), "rb") as f:
            json_file = json.load(f)
            for class_id, v in json_file.items():
                self.syn_to_class[v[0]] = int(class_id)
        with open(os.path.join(root, "ILSVRC2012_val_labels.json"), "rb") as f:
            self.val_to_syn = json.load(f)
        samples_dir = os.path.join(root, "ILSVRC/Data/CLS-LOC", split)
        self.samples_lst = os.listdir(samples_dir)

        for entry in self.samples_lst:
            if split == "train":
                syn_id = entry
                target = self.syn_to_class[syn_id]
                syn_folder = os.path.join(samples_dir, syn_id)
                for sample in os.listdir(syn_folder):
                    sample_path = os.path.join(syn_folder, sample)
                    self.samples.append(sample_path)
                    self.targets.append(target)
            elif split == "val":
                syn_id = self.val_to_syn[entry]
                target = self.syn_to_class[syn_id]
                sample_path = os.path.join(samples_dir, entry)
                self.samples.append(sample_path)
                self.targets.append(target)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x = Image.open(self.samples[idx]).convert("RGB")
        if self.transform:
            x = self.transform(x)
        return x, self.targets[idx]


def prepare_imagenet_subset(root, split='val', n_samples: int = 5000):
    samples_dir = os.path.join(root, "ILSVRC/Data/CLS-LOC", split)
    data_list = np.array(os.listdir(samples_dir))

    np.random.seed(0)
    subset = np.random.choice(data_list, replace=False, size=n_samples)
    subset_dir = Path(os.path.dirname(subsets.__file__))
    np.savetxt(subset_dir / f'imagenet-{n_samples}-{split}.txt', subset, fmt='%s')


def load_imagenet(root: Union[str, Path],
                  split: str = 'val',
                  transform=None,
                  n_samples: Optional[int] = 5000) -> Dataset:
    data = ImageNetKaggle(root=root, split='val', transform=transform)
    if n_samples is None:
        return data

    subset_names_file = Path(os.path.dirname(subsets.__file__)) / f'imagenet-{n_samples}-{split}.txt'
    if not subset_names_file.exists():
        prepare_imagenet_subset(root, split=data.split, n_samples=n_samples)
    subset_names = np.loadtxt(subset_names_file, dtype=str)
    subset_indices = [i for i, file in enumerate(data.samples_lst) if file in subset_names]
    return Subset(dataset=data, indices=subset_indices)
