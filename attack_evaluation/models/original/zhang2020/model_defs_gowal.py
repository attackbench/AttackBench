"""
This code has been taken from the authors' repository.
https://github.com/huanzhang12/CROWN-IBP
"""
import torch.nn as nn


def IBP_large(in_ch, in_dim, linear_size=512):
    model = nn.Sequential(
        nn.Conv2d(in_ch, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 64, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(64, 128, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(128, 128, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(128, 128, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear((in_dim // 2) * (in_dim // 2) * 128, linear_size),
        nn.ReLU(),
        nn.Linear(linear_size, 10)
    )
    return model


def IBP_debug(in_ch, in_dim, linear_size=512):
    model = nn.Sequential(
        nn.Conv2d(1, 1, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(1, 1, 3, stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear((in_dim // 4) * (in_dim // 4) * 1, 10),
    )
    return model


def model_cnn_10layer(in_ch, in_dim, width):
    model = nn.Sequential(
        # input 32*32*3
        nn.Conv2d(in_ch, 4 * width, 3, stride=1, padding=1),
        nn.ReLU(),
        # input 32*32*4
        nn.Conv2d(4 * width, 8 * width, 2, stride=2, padding=0),
        nn.ReLU(),
        # input 16*16*8
        nn.Conv2d(8 * width, 8 * width, 3, stride=1, padding=1),
        nn.ReLU(),
        # input 16*16*8
        nn.Conv2d(8 * width, 16 * width, 2, stride=2, padding=0),
        nn.ReLU(),
        # input 8*8*16
        nn.Conv2d(16 * width, 16 * width, 3, stride=1, padding=1),
        nn.ReLU(),
        # input 8*8*16
        nn.Conv2d(16 * width, 32 * width, 2, stride=2, padding=0),
        nn.ReLU(),
        # input 4*4*32
        nn.Conv2d(32 * width, 32 * width, 3, stride=1, padding=1),
        nn.ReLU(),
        # input 4*4*32
        nn.Conv2d(32 * width, 64 * width, 2, stride=2, padding=0),
        nn.ReLU(),
        # input 2*2*64
        nn.Flatten(),
        nn.Linear(2 * 2 * 64 * width, 10)
    )
    return model


def model_cnn_4layer(in_ch, in_dim, width, linear_size):
    model = nn.Sequential(
        nn.Conv2d(in_ch, 4 * width, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(4 * width, 4 * width, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Conv2d(4 * width, 8 * width, 3, stride=1, padding=1),
        nn.ReLU(),
        nn.Conv2d(8 * width, 8 * width, 4, stride=2, padding=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(8 * width * (in_dim // 4) * (in_dim // 4), linear_size),
        nn.ReLU(),
        nn.Linear(linear_size, linear_size),
        nn.ReLU(),
        nn.Linear(linear_size, 10)
    )
    return model
