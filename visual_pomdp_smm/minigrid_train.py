import os
from torch.utils.tensorboard import SummaryWriter
import torch
# import torch.nn as nn

from tqdm.auto import tqdm
from datetime import datetime

from visual_pomdp_smm.minigrid_utils import (
    Autoencoder, MinigridDatasetParallel, VariationalAutoencoder,
    MinigridDataset, latent_dims,
    input_dims, hidden_size, batch_size,
    epochs, train_set_ratio, in_channels, learning_rate, maximum_gradient
    )

torch.manual_seed(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
save_folder_name = "save"
if not os.path.exists(save_folder_name):
    os.makedirs(save_folder_name)


def train_ae(
        autoencoder, train_dataset, test_dataset,
        epochs=20, log_name="AE"):
    opt = torch.optim.AdamW(autoencoder.parameters(), lr=learning_rate)
    filename_date = str(
        datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-2])

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
            torch.nn.utils.clip_grad_norm_(
                autoencoder.parameters(), maximum_gradient)
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
        autoencoder, save_folder_name + "/" + log_name + "_" +
        filename_date + ".torch")
    return autoencoder


def train_vae(
        autoencoder, train_dataset, test_dataset,
        epochs=20, log_name="VAE"):
    opt = torch.optim.AdamW(autoencoder.parameters(), lr=learning_rate)
    filename_date = str(
        datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-2])

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
            torch.nn.utils.clip_grad_norm_(
                autoencoder.parameters(), maximum_gradient)
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
        autoencoder, save_folder_name + "/" + log_name + "_" +
        filename_date + ".torch")
    return autoencoder


def main_minigrid_ae():

    train_data = MinigridDataset(
        "data/", "train", image_size=input_dims,
        train_set_ratio=train_set_ratio, use_cache=False)
    test_data = MinigridDataset(
        "data/", "test", image_size=input_dims,
        train_set_ratio=train_set_ratio, use_cache=False)

    train_dataset = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True)
    test_dataset = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True)

    autoencoder = Autoencoder(
        input_dims, latent_dims,
        hidden_size, in_channels)
    autoencoder = MinigridDatasetParallel(autoencoder).to(device)
    autoencoder = train_ae(
        autoencoder, train_dataset, test_dataset,
        epochs=epochs, log_name="minigrid_AE")


def main_minigrid_vae():

    train_data = MinigridDataset(
        "data/", "train", image_size=input_dims,
        train_set_ratio=train_set_ratio, use_cache=True)
    test_data = MinigridDataset(
        "data/", "test", image_size=input_dims,
        train_set_ratio=train_set_ratio, use_cache=True)

    train_dataset = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True)
    test_dataset = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True)

    vae = VariationalAutoencoder(
        input_dims, latent_dims,
        hidden_size, in_channels)

    vae = MinigridDatasetParallel(vae).to(device)
    vae = train_vae(
        vae, train_dataset, test_dataset,
        epochs=epochs, log_name="minigrid_VAE")


if __name__ == "__main__":
    main_minigrid_ae()
    # main_minigrid_vae()
