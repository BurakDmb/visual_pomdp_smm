import os
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import json


from tqdm.auto import tqdm
from datetime import datetime
from visual_pomdp_smm.minigrid_utils import (
    MinigridDataset, MinigridMemoryDataset)

from pomdp_tmaze_baselines.utils.AE import Autoencoder, VariationalAutoencoder
from pomdp_tmaze_baselines.utils.AE import ConvAutoencoder
from pomdp_tmaze_baselines.utils.AE import ConvBinaryAutoencoder


torch.manual_seed(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
save_folder_name = "save"
if not os.path.exists(save_folder_name):
    os.makedirs(save_folder_name)


def saveModelWithParams(autoencoder, log_name, filename_date, params):
    save_path = (
        save_folder_name + "/" +
        log_name + "_" + filename_date)
    params['save_path'] = save_path
    torch.save(
        autoencoder, save_path + ".torch")
    with open(save_path + ".json", 'w') as params_file:
        params_file.write(json.dumps(params))


def train_ae_binary(
        autoencoder, train_dataset, test_dataset, params,
        epochs=20, log_name="AE"):
    opt = torch.optim.AdamW(
        autoencoder.parameters(), lr=params['learning_rate'])
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
            x_hat, x_latent = autoencoder(x)
            loss = (
                torch.mean((x - x_hat)**2) +
                params['lambda']*torch.mean(
                    torch.mean(
                        torch.minimum(
                            (x_latent)**2,
                            (1-x_latent)**2), 1), 0))
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                autoencoder.parameters(), params['maximum_gradient'])
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
                x_hat, x_latent = autoencoder(x)
                loss = (
                    torch.mean((x - x_hat)**2) +
                    params['lambda']*torch.mean(
                        torch.mean(
                            torch.minimum(
                                (x_latent)**2,
                                (1-x_latent)**2), 1), 0))
                total_test_loss += loss.item()

        writer.add_scalar(
            "AvgLossPerEpoch/test", total_test_loss/len(test_dataset), epoch)
        writer.flush()
    saveModelWithParams(autoencoder, log_name, filename_date, params)
    # torch.save(
    #     autoencoder, save_folder_name + "/" + log_name + "_" +
    #     filename_date + ".torch")
    return autoencoder


def train_ae(
        autoencoder, train_dataset, test_dataset, params,
        epochs=20, log_name="AE"):
    opt = torch.optim.AdamW(
        autoencoder.parameters(), lr=params['learning_rate'])
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
            x_hat, _ = autoencoder(x)
            loss = ((x - x_hat)**2).sum()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                autoencoder.parameters(), params['maximum_gradient'])
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
                x_hat, _ = autoencoder(x)
                loss = ((x - x_hat)**2).sum()
                total_test_loss += loss.item()

        writer.add_scalar(
            "AvgLossPerEpoch/test", total_test_loss/len(test_dataset), epoch)
        writer.flush()
    saveModelWithParams(autoencoder, log_name, filename_date, params)
    # torch.save(
    #     autoencoder, save_folder_name + "/" + log_name + "_" +
    #     filename_date + ".torch")
    return autoencoder


def train_vae(
        autoencoder, train_dataset, test_dataset, params,
        epochs=20, log_name="VAE"):
    opt = torch.optim.AdamW(
        autoencoder.parameters(), lr=params['learning_rate'])
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
            x_hat, _ = autoencoder(x)
            loss = ((x - x_hat)**2).sum() + autoencoder.encoder.kl
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                autoencoder.parameters(), params['maximum_gradient'])
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
                x_hat, _ = autoencoder(x)
                loss = ((x - x_hat)**2).sum()
                total_test_loss += loss.item()

        writer.add_scalar(
            "AvgLossPerEpoch/test", total_test_loss/len(test_dataset), epoch)
        writer.flush()
    saveModelWithParams(autoencoder, log_name, filename_date, params)
    # torch.save(
    #     autoencoder, save_folder_name + "/" + log_name + "_" +
    #     filename_date + ".torch")
    return autoencoder


def main_minigrid_ae(params):

    # input_dims, hidden_size, batch_size,
    # epochs, train_set_ratio, in_channels, learning_rate, maximum_gradient

    train_data = MinigridDataset(
        "data/", "train", image_size=params['input_dims'],
        train_set_ratio=params['train_set_ratio'], use_cache=False)
    test_data = MinigridDataset(
        "data/", "test", image_size=params['input_dims'],
        train_set_ratio=params['train_set_ratio'], use_cache=False)

    train_dataset = torch.utils.data.DataLoader(
        train_data, batch_size=params['batch_size'], shuffle=True,
        num_workers=4, pin_memory=True)
    test_dataset = torch.utils.data.DataLoader(
        test_data, batch_size=params['batch_size'], shuffle=True,
        num_workers=4, pin_memory=True)

    autoencoder = Autoencoder(
        params['input_dims'], params['latent_dims'],
        params['hidden_size'], params['in_channels'],
        params['kernel_size'], params['padding'],
        params['dilation'], params['conv_hidden_size'],
        params['conv1_stride'], params['maxpool_stride'])
    autoencoder = nn.DataParallel(autoencoder).to(device)
    autoencoder = train_ae(
        autoencoder, train_dataset, test_dataset, params,
        epochs=params['epochs'], log_name="minigrid_AE")


def main_minigrid_memory_binary_ae(params):

    # input_dims, hidden_size, batch_size,
    # epochs, train_set_ratio, in_channels, learning_rate, maximum_gradient

    train_data = MinigridMemoryDataset(
        "data/", "train", image_size=params['input_dims'],
        train_set_ratio=params['train_set_ratio'], use_cache=False)
    test_data = MinigridMemoryDataset(
        "data/", "test", image_size=params['input_dims'],
        train_set_ratio=params['train_set_ratio'], use_cache=False)

    train_dataset = torch.utils.data.DataLoader(
        train_data, batch_size=params['batch_size'], shuffle=True,
        num_workers=1, pin_memory=True)
    test_dataset = torch.utils.data.DataLoader(
        test_data, batch_size=params['batch_size'], shuffle=True,
        num_workers=1, pin_memory=True)

    autoencoder = ConvBinaryAutoencoder(
        params['input_dims'], params['latent_dims'],
        params['hidden_size'], params['in_channels'],
        params['kernel_size'], params['padding'],
        params['dilation'], params['conv_hidden_size'],
        params['conv1_stride'], params['maxpool_stride'])
    autoencoder = nn.DataParallel(autoencoder).to(device)
    autoencoder = train_ae_binary(
        autoencoder, train_dataset, test_dataset, params,
        epochs=params['epochs'], log_name="minigrid_memory_binary_AE")


def main_minigrid_memory_ae(params):

    # input_dims, hidden_size, batch_size,
    # epochs, train_set_ratio, in_channels, learning_rate, maximum_gradient

    train_data = MinigridMemoryDataset(
        "data/", "train", image_size=params['input_dims'],
        train_set_ratio=params['train_set_ratio'], use_cache=False)
    test_data = MinigridMemoryDataset(
        "data/", "test", image_size=params['input_dims'],
        train_set_ratio=params['train_set_ratio'], use_cache=False)

    train_dataset = torch.utils.data.DataLoader(
        train_data, batch_size=params['batch_size'], shuffle=True,
        num_workers=4, pin_memory=True)
    test_dataset = torch.utils.data.DataLoader(
        test_data, batch_size=params['batch_size'], shuffle=True,
        num_workers=4, pin_memory=True)

    autoencoder = ConvAutoencoder(
        params['input_dims'], params['latent_dims'],
        params['hidden_size'], params['in_channels'],
        params['kernel_size'], params['padding'],
        params['dilation'], params['conv_hidden_size'],
        params['conv1_stride'], params['maxpool_stride'])
    autoencoder = nn.DataParallel(autoencoder).to(device)
    autoencoder = train_ae(
        autoencoder, train_dataset, test_dataset, params,
        epochs=params['epochs'], log_name="minigrid_memory_AE")


def main_minigrid_vae(params):

    train_data = MinigridDataset(
        "data/", "train", image_size=params['input_dims'],
        train_set_ratio=params['train_set_ratio'], use_cache=True)
    test_data = MinigridDataset(
        "data/", "test", image_size=params['input_dims'],
        train_set_ratio=params['train_set_ratio'], use_cache=True)

    train_dataset = torch.utils.data.DataLoader(
        train_data, batch_size=params['batch_size'], shuffle=True,
        num_workers=4, pin_memory=True)
    test_dataset = torch.utils.data.DataLoader(
        test_data, batch_size=params['batch_size'], shuffle=True,
        num_workers=4, pin_memory=True)

    vae = VariationalAutoencoder(
        params['input_dims'], params['latent_dims'],
        params['hidden_size'], params['in_channels'],
        params['kernel_size'], params['padding'],
        params['dilation'], params['conv_hidden_size'],
        params['conv1_stride'], params['maxpool_stride'])

    vae = nn.DataParallel(vae).to(device)
    vae = train_vae(
        vae, train_dataset, test_dataset, params,
        epochs=params['epochs'], log_name="minigrid_VAE")


if __name__ == "__main__":
    from visual_pomdp_smm.minigrid_params import params_list
    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)
    processes = []
    for params in params_list:
        # main_minigrid_memory_ae(params)
        # main_minigrid_ae(params)
        # main_minigrid_vae(params)
        p = mp.Process(
            # target=main_minigrid_memory_ae,
            target=main_minigrid_memory_binary_ae,
            # target=main_minigrid_ae,
            # target=main_minigrid_vae,
            args=(params,))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
