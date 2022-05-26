from torch.utils.tensorboard import SummaryWriter
import torch
# import torch.nn as nn
# # import torch.nn.functional as F
# import torch.utils
# import torch.distributions
import torchvision
from tqdm.auto import tqdm
from datetime import datetime
# from pathlib import Path
# import numpy as np
import matplotlib.pyplot as plt

from minigrid_utils import Autoencoder, VariationalAutoencoder, MinigridDataset


plt.rcParams['figure.dpi'] = 200
torch.manual_seed(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def plot_mnist_latent(autoencoder, data, num_batches=100):
    with torch.no_grad():
        plt.figure()

        for i, (x, y) in enumerate(data):
            z = autoencoder.encoder(x.to(device))
            z = z.to('cpu').detach().numpy()
            plt.scatter(z[:, 0], z[:, 1], c=y, cmap='tab10')

        plt.colorbar()


def train_ae(
        autoencoder, train_dataset, test_dataset,
        epochs=20, log_name="AE"):
    opt = torch.optim.AdamW(autoencoder.parameters())
    filename_date = str(datetime.utcnow().strftime('%H_%M_%S_%f')[:-2])

    writer = SummaryWriter(
        "./logs/"+log_name+"/"+filename_date + "/",
        filename_suffix='FC_NN_Last')

    for epoch in tqdm(range(epochs)):
        # Train
        total_training_loss = 0
        for batch_idx, (x, y) in enumerate(train_dataset):
            x = x.to(device)
            opt.zero_grad(set_to_none=True)
            x_hat = autoencoder(x)
            loss = ((x - x_hat)**2).sum()
            loss.backward()
            opt.step()
            total_training_loss += loss.item()

        writer.add_scalar(
            "AvgLossPerEpoch/train",
            total_training_loss/len(train_dataset), epoch)

        # Test
        total_test_loss = 0
        autoencoder.eval()
        with torch.no_grad():
            # for x, y in test_dataset:
            for batch_idx, (x, y) in enumerate(test_dataset):
                x = x.to(device)
                x_hat = autoencoder(x)
                loss = ((x - x_hat)**2).sum()
                total_test_loss += loss.item()

        writer.add_scalar(
            "AvgLossPerEpoch/test", total_test_loss/len(test_dataset), epoch)
        writer.flush()
    torch.save(
        autoencoder, "save/" + log_name + "_" +
        filename_date + ".torch")
    return autoencoder


def train_vae(
        autoencoder, train_dataset, test_dataset,
        epochs=20, log_name="VAE"):
    opt = torch.optim.AdamW(autoencoder.parameters())
    filename_date = str(datetime.utcnow().strftime('%H_%M_%S_%f')[:-2])

    writer = SummaryWriter(
        "./logs/"+log_name+"/"+filename_date + "/",
        filename_suffix='FC_NN_Last')

    for epoch in tqdm(range(epochs)):
        # Train
        total_training_loss = 0
        for batch_idx, (x, y) in enumerate(train_dataset):
            x = x.to(device)
            opt.zero_grad(set_to_none=True)
            x_hat = autoencoder(x)
            loss = ((x - x_hat)**2).sum() + autoencoder.encoder.kl
            loss.backward()
            opt.step()
            total_training_loss += loss.item()

        writer.add_scalar(
            "AvgLossPerEpoch/train",
            total_training_loss/len(train_dataset), epoch)

        # Test
        total_test_loss = 0
        autoencoder.eval()
        with torch.no_grad():
            # for x, y in test_dataset:
            for batch_idx, (x, y) in enumerate(test_dataset):
                x = x.to(device)
                opt.zero_grad()
                x_hat = autoencoder(x)
                loss = ((x - x_hat)**2).sum()
                total_test_loss += loss.item()

        writer.add_scalar(
            "AvgLossPerEpoch/test", total_test_loss/len(test_dataset), epoch)
        writer.flush()
    torch.save(
        autoencoder, "save/" + log_name + "_" +
        filename_date + ".torch")
    return autoencoder


def main_mnist_ae():
    latent_dims = 2
    input_dims = 28
    hidden_size = 128
    batch_size = 512
    epochs = 20
    train_set_ratio = 0.8
    data = torchvision.datasets.MNIST(
        './data',
        transform=torchvision.transforms.ToTensor(),
        download=True)
    in_channels = 1

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
    latent_dims = 2
    input_dims = 28
    hidden_size = 128
    batch_size = 512
    epochs = 20
    train_set_ratio = 0.8
    in_channels = 1

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


def main_minigrid_ae():
    latent_dims = 30
    input_dims = 48
    hidden_size = 128
    batch_size = 512
    epochs = 300
    train_set_ratio = 0.8
    in_channels = 3

    train_data = MinigridDataset(
        "data/", "train", image_size=48, train_set_ratio=train_set_ratio)
    test_data = MinigridDataset(
        "data/", "test", image_size=48, train_set_ratio=train_set_ratio)

    train_dataset = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True,
        num_workers=1, pin_memory=True)
    test_dataset = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=True,
        num_workers=1, pin_memory=True)

    autoencoder = Autoencoder(
        input_dims, latent_dims,
        hidden_size, in_channels).to(device)
    autoencoder = train_ae(
        autoencoder, train_dataset, test_dataset,
        epochs=epochs, log_name="minigrid_AE")


def main_minigrid_vae():
    latent_dims = 30
    input_dims = 48
    hidden_size = 128
    batch_size = 512
    epochs = 300
    train_set_ratio = 0.8
    in_channels = 3

    train_data = MinigridDataset(
        "data/", "train", image_size=48, train_set_ratio=train_set_ratio)
    test_data = MinigridDataset(
        "data/", "test", image_size=48, train_set_ratio=train_set_ratio)

    train_dataset = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True,
        num_workers=1, pin_memory=True)
    test_dataset = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=True,
        num_workers=1, pin_memory=True)

    vae = VariationalAutoencoder(
        input_dims, latent_dims,
        hidden_size, in_channels).to(device)
    vae = train_vae(
        vae, train_dataset, test_dataset,
        epochs=epochs, log_name="minigrid_VAE")


if __name__ == "__main__":
    # main_mnist_ae()
    # main_mnist_vae()
    main_minigrid_ae()
    main_minigrid_vae()
    # plt.show()
