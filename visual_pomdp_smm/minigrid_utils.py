import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


plt.rcParams['figure.dpi'] = 200
torch.manual_seed(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

latent_dims = 3
input_dims = 48
hidden_size = 200
batch_size = 64
epochs = 50
train_set_ratio = 0.8
in_channels = 3
learning_rate = 1e-3
maximum_gradient = 100


# Source: https://avandekleut.github.io/vae/
class Encoder(nn.Module):
    def __init__(self, input_dims, latent_dims, hidden_size, in_channels):
        super(Encoder, self).__init__()
        self.input_dims = input_dims

        kernel_size = 3
        padding = 0
        dilation = 1
        stride = 2

        h_out1 = int(np.floor(((
            input_dims + 2*padding - dilation * (kernel_size-1) - 1)
            / stride) + 1))

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation)

        self.linear1 = nn.Linear(h_out1*h_out1*hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, latent_dims)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        return self.linear2(x)


class Decoder(nn.Module):
    def __init__(self, input_dims, latent_dims, hidden_size, in_channels):
        super(Decoder, self).__init__()
        self.input_dims = input_dims
        self.hidden_size = hidden_size

        kernel_size = 3
        padding = 0
        dilation = 1
        stride = 2

        self.decoder_h_in = (
            stride*(input_dims-1) + 1 - 2*padding
            + dilation * (kernel_size - 1))

        self.linear1 = nn.Linear(latent_dims, hidden_size)
        self.linear2 = nn.Linear(
            hidden_size, self.decoder_h_in*self.decoder_h_in*hidden_size)

        self.conv1 = nn.Conv2d(
            in_channels=hidden_size,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        z = z.reshape((
            -1, self.hidden_size,
            self.decoder_h_in, self.decoder_h_in))
        return self.conv1(z)


class Autoencoder(nn.Module):
    def __init__(self, input_dims, latent_dims, hidden_size, in_channels):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(
            input_dims, latent_dims, hidden_size, in_channels)
        self.decoder = Decoder(
            input_dims, latent_dims, hidden_size, in_channels)

    def forward(self, x):
        self.z = self.encoder(x)
        return self.decoder(self.z)


class VariationalEncoder(nn.Module):
    def __init__(self, input_dims, latent_dims, hidden_size, in_channels):
        super(VariationalEncoder, self).__init__()
        self.input_dims = input_dims

        kernel_size = 3
        padding = 0
        dilation = 1
        stride = 2

        h_out1 = int(np.floor(((
            input_dims + 2*padding - dilation * (kernel_size-1) - 1)
            / stride) + 1))

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=hidden_size,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation)

        self.linear1 = nn.Linear(h_out1*h_out1*hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, latent_dims)
        self.linear3 = nn.Linear(hidden_size, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda()  # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        x = self.conv1(x)
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        mu = self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z


class VariationalAutoencoder(nn.Module):
    def __init__(self, input_dims, latent_dims, hidden_size, in_channels):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(
            input_dims, latent_dims, hidden_size, in_channels)
        self.decoder = Decoder(
            input_dims, latent_dims, hidden_size, in_channels)

    def forward(self, x):
        self.z = self.encoder(x)
        return self.decoder(self.z)


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
