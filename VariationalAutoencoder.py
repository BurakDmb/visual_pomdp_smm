import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions
import torchvision
from tqdm.auto import tqdm

# import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 200
torch.manual_seed(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# TODO: copy minigrid dataset and train it.
# The dimension size of the latent space(3) could not be enough
# for minigrid, try with more dimensions.
# Source: https://avandekleut.github.io/vae/
class Encoder(nn.Module):
    def __init__(self, latent_dims):
        super(Encoder, self).__init__()
        self.linear1 = nn.Linear(784, 512)
        self.linear2 = nn.Linear(512, latent_dims)

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        return self.linear2(x)


class Decoder(nn.Module):
    def __init__(self, latent_dims):
        super(Decoder, self).__init__()
        self.linear1 = nn.Linear(latent_dims, 512)
        self.linear2 = nn.Linear(512, 784)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = torch.sigmoid(self.linear2(z))
        return z.reshape((-1, 1, 28, 28))


class Autoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


class VariationalEncoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(784, 512)
        self.linear2 = nn.Linear(512, latent_dims)
        self.linear3 = nn.Linear(512, latent_dims)

        self.N = torch.distributions.Normal(0, 1)
        self.N.loc = self.N.loc.cuda()  # hack to get sampling on the GPU
        self.N.scale = self.N.scale.cuda()
        self.kl = 0

    def forward(self, x):
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.linear1(x))
        mu = self.linear2(x)
        sigma = torch.exp(self.linear3(x))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z


class VariationalAutoencoder(nn.Module):
    def __init__(self, latent_dims):
        super(VariationalAutoencoder, self).__init__()
        self.encoder = VariationalEncoder(latent_dims)
        self.decoder = Decoder(latent_dims)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z)


def plot_latent(autoencoder, data, num_batches=100):
    with torch.no_grad():
        plt.figure()

        for i, (x, y) in enumerate(data):
            z = autoencoder.encoder(x.to(device))
            z = z.to('cpu').detach().numpy()
            plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')
            if i > num_batches:
                plt.colorbar()
                break


def train_ae(autoencoder, data, epochs=20):
    opt = torch.optim.Adam(autoencoder.parameters())
    for epoch in tqdm(range(epochs)):
        for x, y in data:
            x = x.to(device)
            opt.zero_grad()
            x_hat = autoencoder(x)
            loss = ((x - x_hat)**2).sum()
            loss.backward()
            opt.step()
    return autoencoder


def train_vae(autoencoder, data, epochs=20):
    opt = torch.optim.Adam(autoencoder.parameters())
    for epoch in tqdm(range(epochs)):
        for x, y in data:
            x = x.to(device)
            opt.zero_grad()
            x_hat = autoencoder(x)
            loss = ((x - x_hat)**2).sum() + autoencoder.encoder.kl
            loss.backward()
            opt.step()
    return autoencoder


def main_ae():
    latent_dims = 2
    autoencoder = Autoencoder(latent_dims).to(device)

    data = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(
                './data',
                transform=torchvision.transforms.ToTensor(),
                download=True),
            batch_size=128,
            shuffle=True)

    autoencoder = train_ae(autoencoder, data, epochs=20)
    plot_latent(autoencoder, data)


def main_vae():
    latent_dims = 2
    data = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(
                './data',
                transform=torchvision.transforms.ToTensor(),
                download=True),
            batch_size=128,
            shuffle=True)

    vae = VariationalAutoencoder(latent_dims).to(device)
    vae = train_vae(vae, data, epochs=20)
    plot_latent(vae, data)


if __name__ == "__main__":
    main_ae()
    main_vae()
    plt.show()
