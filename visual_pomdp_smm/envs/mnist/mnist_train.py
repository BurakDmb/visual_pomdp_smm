import os

import matplotlib.pyplot as plt
import torch
import torchvision

from visual_pomdp_smm.training.AE import Autoencoder, VariationalAutoencoder
from visual_pomdp_smm.training.train_utils import train_ae, train_vae

plt.rcParams['figure.dpi'] = 200
# torch.manual_seed(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
save_folder_name = "save"
if not os.path.exists(save_folder_name):
    os.makedirs(save_folder_name)

latent_dims = 2
input_dims = 28
hidden_size = 128
batch_size = 512
epochs = 20
train_set_ratio = 0.8
in_channels = 1


def plot_mnist_latent(autoencoder, data, num_batches=100):
    with torch.no_grad():
        plt.figure()

        for i, (x, y) in enumerate(data):
            z = autoencoder.encoder(x.to(device))
            z = z.to('cpu').detach().numpy()
            plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')

        plt.colorbar()


def main_mnist_ae():

    data = torchvision.datasets.MNIST(
        './data',
        transform=torchvision.transforms.ToTensor(),
        download=True)

    lengths = [
        int(len(data)*train_set_ratio),
        len(data) - int(len(data)*train_set_ratio)]

    train_data, test_data = torch.utils.data.random_split(
        data, lengths)

    train_dataset = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True)
    test_dataset = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True)

    autoencoder = Autoencoder(
        input_dims, latent_dims,
        hidden_size, in_channels).to(device)
    autoencoder = train_ae(
        autoencoder, train_dataset, test_dataset,
        epochs=epochs, log_name="mnist_AE")
    plot_mnist_latent(autoencoder, test_dataset)


def main_mnist_vae():

    data = torchvision.datasets.MNIST(
        './data',
        transform=torchvision.transforms.ToTensor(),
        download=True)

    lengths = [
        int(len(data)*train_set_ratio),
        len(data) - int(len(data)*train_set_ratio)]
    train_data, test_data = torch.utils.data.random_split(
        data, lengths)

    train_dataset = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True)
    test_dataset = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True)

    vae = VariationalAutoencoder(
        input_dims, latent_dims,
        hidden_size, in_channels).to(device)
    vae = train_vae(
        vae, train_dataset, test_dataset,
        epochs=epochs, log_name="mnist_VAE")
    plot_mnist_latent(vae, test_dataset)


if __name__ == "__main__":
    main_mnist_ae()
    main_mnist_vae()
    plt.show()
