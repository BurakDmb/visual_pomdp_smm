import json
import os
from datetime import datetime
from itertools import repeat

import torch
import torch.multiprocessing as mp
import torch.nn as nn
from pomdp_tmaze_baselines.utils.AE import (Autoencoder, ConvAutoencoder,
                                            ConvBinaryAutoencoder,
                                            ConvVariationalAutoencoder,
                                            VariationalAutoencoder)
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from visual_pomdp_smm.envs.minigrid.minigrid_utils import \
    MinigridGenericDataset

# torch.manual_seed(0)
# torch.autograd.set_detect_anomaly(True)

save_folder_name = "save"
torch_folder_name = "save/torch"
json_folder_name = "save/json"
if not os.path.exists(save_folder_name):
    os.makedirs(save_folder_name)
if not os.path.exists(torch_folder_name):
    os.makedirs(torch_folder_name)
if not os.path.exists(json_folder_name):
    os.makedirs(json_folder_name)


def saveModelWithParams(autoencoder, log_name, filename_date, params):
    torch_save_path = (
        save_folder_name + "/torch/" +
        log_name + "_" + filename_date)
    json_save_path = (
        save_folder_name + "/json/" +
        log_name + "_" + filename_date)
    params['save_path'] = torch_save_path
    torch.save(
        autoencoder, torch_save_path + ".torch")
    with open(json_save_path + ".json", 'w') as params_file:
        params_file.write(json.dumps(params))


def train_ae_binary(
        autoencoder, train_dataset, test_dataset, params, device,
        epochs=20, log_name="AE"):
    opt = torch.optim.AdamW(
        autoencoder.parameters(), lr=params['learning_rate'])
    filename_date = str(
        datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-2])

    writer = SummaryWriter(
        "./logs/"+log_name+"/"+filename_date + "/",
        filename_suffix='FC_NN_Last')

    loss_func = nn.L1Loss()
    # loss_func = nn.BCELoss()

    for epoch in tqdm(range(epochs)):
        # Train
        total_training_loss = 0
        for batch_idx, (x, y) in enumerate(train_dataset):
            x = x.to(device)
            opt.zero_grad(set_to_none=True)
            x_hat, x_latent = autoencoder(x)
            loss = (
                loss_func(x_hat, x) +
                params['lambda']*(torch.minimum(
                    (x_latent)**2,
                    (1-x_latent)**2
                    ).sum()/(
                        params['batch_size'] *
                        params['latent_dims']))
                ) / (1+params['lambda'])
            # Dividing by 1+lambda since we are
            # normalizing the whole sum

            # loss = (
            #     (((x-x_hat)**2).sum(dim=(1, 2, 3))/(
            #         params['in_channels'] *
            #         params['input_dims_h'] *
            #         params['input_dims_w'])) +
            #     params['lambda']*(torch.minimum(
            #         (x_latent)**2,
            #         (1-x_latent)**2
            #         ).sum(dim=1)/params['latent_dims'])
            #     ).sum(dim=0)/(params['batch_size']*(1+params['lambda']))
            # Dividing by 1+ lambda since we are also
            # normalizing the second term

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
                loss = loss_func(x_hat, x)
                # loss = (
                #     loss_func(x_hat, x) +
                #     params['lambda']*(torch.minimum(
                #         (x_latent)**2,
                #         (1-x_latent)**2
                #         ).sum()/(
                #             params['batch_size'] *
                #             params['latent_dims']))
                #     ) / (1+params['lambda'])
                # loss = (
                #     (((x-x_hat)**2).sum(dim=(1, 2, 3))/(
                #         params['in_channels'] *
                #         params['input_dims_h'] *
                #         params['input_dims_w'])) +
                #     params['lambda']*(torch.minimum(
                #         (x_latent)**2,
                #         (1-x_latent)**2
                #         ).sum(dim=1)/params['latent_dims'])
                #     ).sum(dim=0)/(params['batch_size']*(1+params['lambda']))
                # Dividing by 1+ lambda since we are also
                # normalizing the second term
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
        autoencoder, train_dataset, test_dataset, params, device,
        epochs=20, log_name="AE"):
    opt = torch.optim.AdamW(
        autoencoder.parameters(), lr=params['learning_rate'])
    filename_date = str(
        datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-2])

    writer = SummaryWriter(
        "./logs/"+log_name+"/"+filename_date + "/",
        filename_suffix='FC_NN_Last')

    loss_func = nn.L1Loss()
    # loss_func = nn.BCELoss()

    for epoch in tqdm(range(epochs)):
        # Train
        total_training_loss = 0
        for batch_idx, (x, y) in enumerate(train_dataset):
            x = x.to(device)
            opt.zero_grad(set_to_none=True)
            x_hat, _ = autoencoder(x)
            loss = loss_func(x_hat, x)
            # loss = ((x - x_hat)**2).sum() / (
            #         params['in_channels'] *
            #         params['input_dims_h'] *
            #         params['input_dims_w'] *
            #         params['batch_size'])
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
                loss = loss_func(x_hat, x)
                # loss = ((x - x_hat)**2).sum() / (
                #     params['in_channels'] *
                #     params['input_dims_h'] *
                #     params['input_dims_w'] *
                #     params['batch_size'])
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
        autoencoder, train_dataset, test_dataset, params, device,
        epochs=20, log_name="VAE"):
    opt = torch.optim.AdamW(
        autoencoder.parameters(), lr=params['learning_rate'])
    filename_date = str(
        datetime.utcnow().strftime('%Y_%m_%d_%H_%M_%S_%f')[:-2])

    writer = SummaryWriter(
        "./logs/"+log_name+"/"+filename_date + "/",
        filename_suffix='FC_NN_Last')

    loss_func = nn.L1Loss()
    # loss_func = nn.BCELoss()

    for epoch in tqdm(range(epochs)):
        # Train
        total_training_loss = 0
        for batch_idx, (x, y) in enumerate(train_dataset):
            x = x.to(device)
            opt.zero_grad(set_to_none=True)
            x_hat, _ = autoencoder(x)
            if hasattr(autoencoder, 'module'):
                loss = loss_func(x_hat, x) + autoencoder.module.encoder.kl
            else:
                loss = loss_func(x_hat, x) + autoencoder.encoder.kl
            # loss = ((x - x_hat)**2).sum() / (
            #     params['in_channels'] *
            #     params['input_dims_h'] *
            #     params['input_dims_w'] *
            #     params['batch_size']
            #     ) + autoencoder.module.encoder.kl
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
                loss = loss_func(x_hat, x)
                # loss = (
                #     loss_func(x_hat, x) + autoencoder.module.encoder.kl)
                # loss = ((x - x_hat)**2).sum() / (
                #     params['in_channels'] *
                #     params['input_dims_h'] *
                #     params['input_dims_w'] *
                #     params['batch_size']
                #     ) + autoencoder.module.encoder.kl
                total_test_loss += loss.item()

        writer.add_scalar(
            "AvgLossPerEpoch/test", total_test_loss/len(test_dataset), epoch)
        writer.flush()
    saveModelWithParams(autoencoder, log_name, filename_date, params)
    # torch.save(
    #     autoencoder, save_folder_name + "/" + log_name + "_" +
    #     filename_date + ".torch")
    return autoencoder


def start_training(params):
    torch.cuda.set_device(params['gpu_id'])
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'

    train_class_str = params['train_class']
    if train_class_str == 'train_ae':
        train_func = train_ae
    elif train_class_str == 'train_vae':
        train_func = train_vae
    elif train_class_str == 'train_ae_binary':
        train_func = train_ae_binary
    else:
        print("Wrong train function string passed, ending execution.")
        exit(1)

    ae_class_str = params['ae_class']
    if ae_class_str == 'Autoencoder':
        ae_class = Autoencoder
    elif ae_class_str == 'VariationalAutoencoder':
        ae_class = VariationalAutoencoder
    elif ae_class_str == 'ConvAutoencoder':
        ae_class = ConvAutoencoder
    elif ae_class_str == 'ConvVariationalAutoencoder':
        ae_class = ConvVariationalAutoencoder
    elif ae_class_str == 'ConvBinaryAutoencoder':
        ae_class = ConvBinaryAutoencoder
    else:
        print("Wrong ae class string passed, ending execution.")
        exit(1)

    # input_dims, hidden_size, batch_size,
    # epochs, train_set_ratio, in_channels, learning_rate, maximum_gradient

    train_data = MinigridGenericDataset(
        "data/", "train",
        image_size_h=params['input_dims_h'],
        image_size_w=params['input_dims_w'],
        train_set_ratio=params['train_set_ratio'],
        dataset_folder_name=params['dataset_folder_name'], use_cache=False)
    test_data = MinigridGenericDataset(
        "data/", "test",
        image_size_h=params['input_dims_h'],
        image_size_w=params['input_dims_w'],
        train_set_ratio=params['train_set_ratio'],
        dataset_folder_name=params['dataset_folder_name'], use_cache=False)

    train_dataset = torch.utils.data.DataLoader(
        train_data, batch_size=params['batch_size'], shuffle=False,
        num_workers=0, pin_memory=False, persistent_workers=False)
    test_dataset = torch.utils.data.DataLoader(
        test_data, batch_size=params['batch_size'], shuffle=False,
        num_workers=0, pin_memory=False, persistent_workers=False)

    autoencoder = ae_class(**params).to(device)
    # autoencoder = nn.DataParallel(autoencoder).to(device)
    autoencoder = train_func(
        autoencoder, train_dataset, test_dataset, params, device,
        epochs=params['epochs'], log_name=params['log_name'])


def mp_pool_function(param, queue):
    gpu_id = queue.get()
    try:
        param['gpu_id'] = gpu_id
        print(
            "Experiment log_name=" + param['log_name'] +
            " (gpu_id=" + str(gpu_id) + ") started.")
        start_training(param)
        print(
            "Experiment log_name=" + param['log_name'] +
            " (gpu_id=" + str(gpu_id) + ") finished.")
    except Exception as e:
        print(e)
    finally:
        queue.put(gpu_id)


def mp_create_experiment_params(params_list, N):
    # Creating experiment parameter sets,
    # while assining each with an unique id.
    experiment_params = []
    for param in params_list:
        for i in range(N):
            param_ = param.copy()
            param_['log_name'] = param_['log_name'] + '_' + str(i)
            experiment_params.append(param_)
    return experiment_params


# For The GPU Job Distribution, stackoverflow has been used.
# https://stackoverflow.com/questions/53422761/distributing-jobs-evenly-across-multiple-gpus-with-multiprocessing-pool
def start_multi_training(params_list, NUM_GPUS, PROC_PER_GPU, N):
    mp.set_start_method('spawn', force=True)
    manager = mp.Manager()
    queue = manager.Queue()
    experiment_params = mp_create_experiment_params(params_list, N)

    # Initialize the queue with the GPU ids
    for gpu_ids in range(NUM_GPUS):
        for _ in range(PROC_PER_GPU):
            queue.put(gpu_ids)

    pool = mp.Pool(processes=PROC_PER_GPU * NUM_GPUS)

    for _ in pool.starmap(mp_pool_function, zip(
            experiment_params, repeat(queue))):
        pass
    pool.close()
    pool.join()
