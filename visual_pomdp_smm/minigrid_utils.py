import torch
# import torch.nn as nn
# import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision

from pathlib import Path
# import numpy as np
import matplotlib.pyplot as plt


plt.rcParams['figure.dpi'] = 200
torch.manual_seed(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

latent_dims = 128
input_dims = 48
hidden_size = 32
batch_size = 64
epochs = 1000
train_set_ratio = 0.8
in_channels = 3
learning_rate = 1e-4
maximum_gradient = 1e7

kernel_size = 4
padding = 3
dilation = 1
conv_hidden_size = 32
conv1_stride = 6
maxpool_stride = 1


class MinigridDataset(torch.utils.data.Dataset):
    def __init__(
            self, data_path, split, image_size, train_set_ratio,
            use_cache=False, **kwargs):

        self.data_dir = Path(data_path) / "Minigrid"
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(image_size),
            torchvision.transforms.ToTensor(), ])

        imgs = sorted([
            f for f in self.data_dir.iterdir() if f.suffix == '.png'])

        self.imgs = imgs[
            :int(len(imgs) * train_set_ratio)] if split == "train" else imgs[
            int(len(imgs) * train_set_ratio):]

        self.cached_data = []
        self.use_cache = use_cache
        if self.use_cache:
            # caching for speedup
            for i in range(len(self.imgs)):
                img = torchvision.datasets.folder.default_loader(self.imgs[i])
                if self.transforms is not None:
                    img = self.transforms(img)
                self.cached_data.append(img)
            self.cached_data = torch.stack(self.cached_data)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        if not self.use_cache:
            img = torchvision.datasets.folder.default_loader(self.imgs[idx])
            if self.transforms is not None:
                img = self.transforms(img)
        else:
            img = self.cached_data[idx]

        return img, 0.0  # dummy data_y to prevent breaking


class MinigridMemoryDataset(torch.utils.data.Dataset):
    def __init__(
            self, data_path, split, image_size, train_set_ratio,
            use_cache=False, **kwargs):

        self.data_dir = Path(data_path) / "MinigridKey"
        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize(image_size),
            torchvision.transforms.ToTensor(), ])

        if split == "key":
            imgs = sorted([
                f for f in self.data_dir.iterdir()
                if f.match('minigridkey_key*.png')])
            self.imgs = imgs
        else:
            imgs = sorted([
                f for f in self.data_dir.iterdir()
                if f.match('minigridkey_*.png')])

            self.imgs = imgs[
                :int(len(imgs) * train_set_ratio)
                ] if split == "train" else imgs[
                int(len(imgs) * train_set_ratio):]

        self.cached_data = []
        self.use_cache = use_cache
        if self.use_cache:
            # caching for speedup
            for i in range(len(self.imgs)):
                img = torchvision.datasets.folder.default_loader(self.imgs[i])
                if self.transforms is not None:
                    img = self.transforms(img)
                self.cached_data.append(img)
            self.cached_data = torch.stack(self.cached_data)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        if not self.use_cache:
            img = torchvision.datasets.folder.default_loader(self.imgs[idx])
            if self.transforms is not None:
                img = self.transforms(img)
        else:
            img = self.cached_data[idx]

        return img, 0.0  # dummy data_y to prevent breaking
