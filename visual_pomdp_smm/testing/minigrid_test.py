import json
import os
import random

# import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm.auto import tqdm

from visual_pomdp_smm.envs.minigrid.minigrid_utils import (
    MinigridDataset, MinigridMemoryUniformDatasetEval,
    MinigridMemoryFullDataset, MinigridMemoryKeyDataset,
    MinigridMemoryUniformDataset,
    MinigridMemoryUniformDatasetNoteval,
    MinigridDynamicObsUniformDataset,
    MinigridDynamicObsUniformDatasetNoteval,
    MinigridDynamicObsUniformDatasetEval)

# plt.rcParams['figure.dpi'] = 200
# matplotlib.use('GTK3Agg')

torch.manual_seed(0)
random.seed(None)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def test_model(
        autoencoder, test_dataset):

    test_losses = []
    total_test_loss = 0
    total_sample_number = 0
    latentArray = []
    lossesArray = []
    autoencoder.eval()

    loss_func = nn.L1Loss(reduction='none')

    with torch.no_grad():
        # for x, y in test_dataset:
        for batch_idx, (x, y) in enumerate(test_dataset):
            total_sample_number += len(x)
            x = x.to(device)
            x_hat, z = autoencoder(x)
            loss_values = loss_func(x, x_hat)
            loss = loss_values.sum()
            losses = loss_values.sum((1, 2, 3)).cpu().numpy().tolist()
            lossesArray.extend(losses)
            total_test_loss += loss.item()
            test_losses.append(loss.item())
            latentArray.extend(z.cpu().numpy().tolist())

    return total_test_loss, total_sample_number,\
        latentArray, test_losses, lossesArray


# TODO: Needs re-training.
def test_minigrid_memory_binary_ae(random_visualize=False):
    test_function(
        'MinigridMemoryFullDataset', 'MinigridMemoryKeyDataset',
        prefix_name='minigrid_memory_binary_AE_2022',
        random_visualize=random_visualize)
    return
    prefix_name = "minigrid_memory_binary_AE_2022"
    dirFiles = os.listdir('save/json')
    prefixed = [filename for filename in dirFiles
                if filename.startswith(prefix_name)]

    prefixed.sort(reverse=True)
    for filename in prefixed:

        with open("save/json/"+filename, 'r') as params_file:
            params = json.loads(params_file.read())
        model_file_name = params['save_path']

        print("Testing for param path: " + params['save_path'])
        with torch.no_grad():
            ae = torch.load(model_file_name+".torch").to(device)
            ae = ae.module
            ae.eval()

            test_data = MinigridMemoryFullDataset(
                "data/", "test",
                image_size=params['input_dims'],
                train_set_ratio=params['train_set_ratio'])

            test_dataset = torch.utils.data.DataLoader(
                test_data, batch_size=256, shuffle=True,
                num_workers=1, pin_memory=True)

            key_data = MinigridMemoryFullDataset(
                "data/", "key",
                image_size=params['input_dims'],
                train_set_ratio=params['train_set_ratio'])

            key_dataset = torch.utils.data.DataLoader(
                key_data, batch_size=256, shuffle=True,
                num_workers=1, pin_memory=True)

            # Calculating loss in batches
            test_loss, test_sample_number, latentArray,\
                test_losses, lossesArray = test_model(ae, test_dataset)

            norm_test_loss = (
                np.array(test_losses) / (
                    1*params['input_dims'] *
                    params['input_dims'] *
                    params['in_channels'] *
                    256)
                )

            print(
                "Test Dataset AvgLoss: ",
                "{:.2e}".format(norm_test_loss.sum()/test_sample_number))
            print(
                "Test Dataset min and max normalized loss. Min: " +
                str(norm_test_loss.min()) +
                ", Max: " +
                str(norm_test_loss.max()))

            # Calculating loss in batches
            key_loss, key_sample_number, keylatentArray,\
                key_losses, keyLossesArray = test_model(ae, key_dataset)
            norm_key_losses = (
                np.array(key_losses) / (
                    1*params['input_dims'] *
                    params['input_dims'] *
                    params['in_channels'] *
                    256)
                )

            print(
                "Key Dataset AvgLoss",
                "{:.2e}".format(norm_key_losses.sum()/key_sample_number))
            print(
                "Key Dataset min and max normalized loss. Min: " +
                str(norm_key_losses.min()) +
                ", Max: " +
                str(norm_key_losses.max()))

            percent_difference = (
                (norm_key_losses.sum()/key_sample_number)
                / (norm_test_loss.sum()/test_sample_number)
                ) * 100 - 100
            print("Percent diff (key/test) (%): ", percent_difference)

            random_data = test_data[
                random.randint(0, len(test_data))]
            random_data_hat, _ = ae(
                torch.unsqueeze(random_data[0], 0).to(device))

            random_data_image = np.uint8(
                random_data[0].cpu().numpy()*255
                ).transpose(1, 2, 0)
            random_data_hat_image = np.uint8(
                random_data_hat[0].cpu().numpy()*255
                ).transpose(1, 2, 0)
            im_orig = Image.fromarray(random_data_image)
            im_generated = Image.fromarray(random_data_hat_image)

            if not os.path.exists("save/results"):
                os.makedirs("save/results")

            im_orig.save(
                "save/results/im_orig_" +
                filename.replace(".json", "") + ".png")
            im_generated.save(
                "save/results/im_generated_" +
                filename.replace(".json", "") + ".png")

            random_key_data = key_data[
                random.randint(0, len(key_data))]
            random_key_data_hat, _ = ae(
                torch.unsqueeze(random_key_data[0], 0).to(device))

            random_key_data_image = np.uint8(
                random_key_data[0].cpu().numpy()*255
                ).transpose(1, 2, 0)
            random_key_data_hat_image = np.uint8(
                random_key_data_hat[0].cpu().numpy()*255
                ).transpose(1, 2, 0)
            im_key_orig = Image.fromarray(random_key_data_image)
            im_key_generated = Image.fromarray(random_key_data_hat_image)

            im_key_orig.save(
                "save/results/im_key_orig_" +
                filename.replace(".json", "") + ".png")
            im_key_generated.save(
                "save/results/im_key_generated_" +
                filename.replace(".json", "") + ".png")
            print("")


def test_minigrid_memory_ae(random_visualize=False):
    test_function(
        'MinigridMemoryFullDataset', 'MinigridMemoryKeyDataset',
        prefix_name='minigrid_memory_AE_2022',
        random_visualize=random_visualize)
    return
    prefix_name = "minigrid_memory_AE_2022"
    dirFiles = os.listdir('save/json')
    prefixed = [filename for filename in dirFiles
                if filename.startswith(prefix_name)]

    prefixed.sort(reverse=True)
    for filename in prefixed:
        with open("save/json/"+filename, 'r') as params_file:
            params = json.loads(params_file.read())
        model_file_name = params['save_path']
        print("Testing for param path: " + params['save_path'])

        with torch.no_grad():
            ae = torch.load(model_file_name+".torch").to(device)
            ae = ae.module
            ae.eval()

            test_data = MinigridMemoryFullDataset(
                "data/", "test",
                image_size=params['input_dims'],
                train_set_ratio=params['train_set_ratio'])

            test_dataset = torch.utils.data.DataLoader(
                test_data, batch_size=256, shuffle=True,
                num_workers=1, pin_memory=True)

            key_data = MinigridMemoryFullDataset(
                "data/", "key",
                image_size=params['input_dims'],
                train_set_ratio=params['train_set_ratio'])

            key_dataset = torch.utils.data.DataLoader(
                key_data, batch_size=256, shuffle=True,
                num_workers=1, pin_memory=True)

            test_loss, test_sample_number, latentArray,\
                test_losses, lossesArray = test_model(ae, test_dataset)

            norm_test_loss = (
                np.array(test_losses) / (
                    1*params['input_dims'] *
                    params['input_dims'] *
                    params['in_channels'])
                )

            print(
                "Test Dataset AvgLoss: ",
                "{:.2e}".format(norm_test_loss.sum()/test_sample_number))
            print(
                "Test Dataset min and max normalized loss. Min: " +
                str(norm_test_loss.min()) +
                ", Max: " +
                str(norm_test_loss.max()))

            key_loss, key_sample_number, keylatentArray,\
                key_losses, keyLossesArray = test_model(ae, key_dataset)
            norm_key_losses = (
                np.array(key_losses) / (
                    1*params['input_dims'] *
                    params['input_dims'] *
                    params['in_channels'])
                )

            print(
                "Key Dataset AvgLoss",
                "{:.2e}".format(norm_key_losses.sum()/key_sample_number))
            print(
                "Key Dataset min and max normalized loss. Min: " +
                str(norm_key_losses.min()) +
                ", Max: " +
                str(norm_key_losses.max()))

            percent_difference = (
                (norm_key_losses.sum()/key_sample_number)
                / (norm_test_loss.sum()/test_sample_number)
                ) * 100 - 100
            print("Percent diff (key/test) (%): ", percent_difference)

            random_data = test_data[
                random.randint(0, len(test_data))]
            random_data_hat, _ = ae(
                torch.unsqueeze(random_data[0], 0).to(device))

            random_data_image = np.uint8(
                random_data[0].cpu().numpy()*255
                ).transpose(1, 2, 0)
            random_data_hat_image = np.uint8(
                random_data_hat[0].cpu().numpy()*255
                ).transpose(1, 2, 0)
            im_orig = Image.fromarray(random_data_image)
            im_generated = Image.fromarray(random_data_hat_image)

            # im_orig.show("im_orig.png")
            # im_generated.show("im_generated.png")

            if not os.path.exists("save/results"):
                os.makedirs("save/results")

            im_orig.save(
                "save/results/im_orig_" +
                filename.replace(".json", "") + ".png")
            im_generated.save(
                "save/results/im_generated_" +
                filename.replace(".json", "") + ".png")

            random_key_data = key_data[
                random.randint(0, len(key_data))]
            random_key_data_hat, _ = ae(
                torch.unsqueeze(random_key_data[0], 0).to(device))

            random_key_data_image = np.uint8(
                random_key_data[0].cpu().numpy()*255
                ).transpose(1, 2, 0)
            random_key_data_hat_image = np.uint8(
                random_key_data_hat[0].cpu().numpy()*255
                ).transpose(1, 2, 0)
            im_key_orig = Image.fromarray(random_key_data_image)
            im_key_generated = Image.fromarray(random_key_data_hat_image)

            im_key_orig.save(
                "save/results/im_key_orig_" +
                filename.replace(".json", "") + ".png")
            im_key_generated.save(
                "save/results/im_key_generated_" +
                filename.replace(".json", "") + ".png")
            print("")


def test_minigrid_ae(params, random_visualize=False):
    print("Testing for param path: " + params['save_path'])
    prefix_name = "minigrid_AE_2022"
    dirFiles = os.listdir('save')
    prefixed = [filename for filename in dirFiles
                if filename.startswith(prefix_name)]

    prefixed.sort(reverse=True)
    model_file_name = prefixed[0]
    with torch.no_grad():
        ae = torch.load("save/" + model_file_name).to(device)
        ae = ae.module
        ae.eval()

        test_data = MinigridDataset(
            "data/", "test",
            image_size=params['input_dims'],
            train_set_ratio=params['train_set_ratio'])

        test_dataset = torch.utils.data.DataLoader(
            test_data, batch_size=params['batch_size'], shuffle=True,
            num_workers=1, pin_memory=True)

        test_loss, _, latentArray,\
            test_losses, lossesArray = test_model(ae, test_dataset)
        print(test_loss)

        random_data = test_data[
            random.randint(0, len(test_data))]
        random_data_hat, _ = ae(torch.unsqueeze(random_data[0], 0).to(device))

        random_data_image = np.uint8(
            random_data[0].cpu().numpy()*255
            ).transpose(1, 2, 0)
        random_data_hat_image = np.uint8(
            random_data_hat[0].cpu().numpy()*255
            ).transpose(1, 2, 0)
        im_orig = Image.fromarray(random_data_image)
        im_generated = Image.fromarray(random_data_hat_image)
        im_orig.show()
        im_generated.show()
        im_orig.save("results/im_orig.png")
        im_generated.save("results/im_generated.png")
        scatterDatasetLatent(latentArray)


def test_minigrid_vae(params, random_visualize=False):
    print("Testing for param path: " + params['save_path'])
    prefix_name = "minigrid_VAE_2022"
    dirFiles = os.listdir('save')
    prefixed = [filename for filename in dirFiles
                if filename.startswith(prefix_name)]

    prefixed.sort(reverse=True)
    model_file_name = prefixed[0]
    with torch.no_grad():

        vae = torch.load("save/" + model_file_name).to(device)
        vae = vae.module
        vae.eval()

        test_data = MinigridDataset(
            "data/", "test",
            image_size=params['input_dims'],
            train_set_ratio=params['train_set_ratio'],
            use_cache=False)

        test_dataset = torch.utils.data.DataLoader(
            test_data, batch_size=params['batch_size'], shuffle=True,
            num_workers=1, pin_memory=True)

        test_loss, _, latentArray,\
            test_losses, lossesArray = test_model(vae, test_dataset)
        print(test_loss)
        random_data = test_data[
            random.randint(0, len(test_data))]
        random_data_hat, _ = vae(torch.unsqueeze(random_data[0], 0).to(device))

        random_data_image = np.uint8(
            random_data[0].cpu().numpy()*255
            ).transpose(1, 2, 0)
        random_data_hat_image = np.uint8(
            random_data_hat[0].cpu().numpy()*255
            ).transpose(1, 2, 0)
        im_orig = Image.fromarray(random_data_image)
        im_generated = Image.fromarray(random_data_hat_image)
        im_orig.show()
        im_generated.show()


def test_function(
        test_dataset_class_str, eval_dataset_class_str,
        prefix_name, random_visualize=False):

    if test_dataset_class_str == 'MinigridMemoryFullDataset':
        test_dataset_class = MinigridMemoryFullDataset
    elif test_dataset_class_str == 'MinigridMemoryUniformDataset':
        test_dataset_class = MinigridMemoryUniformDataset
    elif test_dataset_class_str == 'MinigridMemoryUniformDatasetNoteval':
        test_dataset_class = MinigridMemoryUniformDatasetNoteval
    elif test_dataset_class_str == 'MinigridDynamicObsUniformDataset':
        test_dataset_class = MinigridDynamicObsUniformDataset
    elif test_dataset_class_str == 'MinigridDynamicObsUniformDatasetNoteval':
        test_dataset_class = MinigridDynamicObsUniformDatasetNoteval
    else:
        print(
            "Invalid test dataset class string, " +
            "ending execution of the program.")
        exit(1)
    if eval_dataset_class_str == 'MinigridMemoryKeyDataset':
        eval_dataset_class = MinigridMemoryKeyDataset
    elif eval_dataset_class_str == 'MinigridMemoryUniformDatasetEval':
        eval_dataset_class = MinigridMemoryUniformDatasetEval
    elif eval_dataset_class_str == 'MinigridDynamicObsUniformDatasetEval':
        eval_dataset_class = MinigridDynamicObsUniformDatasetEval
    else:
        print(
            "Invalid eval dataset class string, " +
            "ending execution of the program.")
        exit(1)

    dirFiles = os.listdir('save/json')
    prefixed = [filename for filename in dirFiles
                if filename.startswith(prefix_name)]

    prefixed.sort(reverse=True)
    for filename in prefixed:
        with open("save/json/"+filename, 'r') as params_file:
            params = json.loads(params_file.read())
        model_file_name = params['save_path']
        print("Testing for param path: " + params['save_path'])

        with torch.no_grad():
            ae = torch.load(model_file_name+".torch").to(device)
            ae = ae.module
            ae.eval()

            test_data = test_dataset_class(
                "data/", "test",
                image_size=params['input_dims'],
                train_set_ratio=params['train_set_ratio'],
                use_cache=False)

            test_dataset = torch.utils.data.DataLoader(
                test_data, batch_size=128, shuffle=True,
                num_workers=0, pin_memory=True)

            eval_class_data = eval_dataset_class(
                "data/", "",
                image_size=params['input_dims'],
                train_set_ratio=params['train_set_ratio'],
                use_cache=False)

            eval_class_dataset = torch.utils.data.DataLoader(
                eval_class_data, batch_size=128, shuffle=True,
                num_workers=0, pin_memory=True)

            total_test_loss, test_sample_number, latentArrays,\
                test_losses, testLossesArray = test_model(ae, test_dataset)

            norm_test_losses = (
                np.array(testLossesArray) / (
                    params['input_dims'] *
                    params['input_dims'] *
                    params['in_channels'])
                )
            print(
                "Test Dataset AvgLoss: ",
                "{:.2e}".format(norm_test_losses.sum()/test_sample_number))
            print(
                "Test Dataset min and max normalized loss. Min: " +
                "{:.2e}".format(norm_test_losses.min()) +
                ", Max: " +
                "{:.2e}".format(norm_test_losses.max()))

            total_eval_loss, eval_sample_number, evalLatentArrays,\
                eval_losses, evalLossesArray = test_model(
                    ae, eval_class_dataset)
            norm_eval_losses = (
                np.array(evalLossesArray) / (
                    params['input_dims'] *
                    params['input_dims'] *
                    params['in_channels'])
                )

            print(
                "Eval Class Dataset AvgLoss",
                "{:.2e}".format(norm_eval_losses.sum()/eval_sample_number))
            print(
                "Eval Class Dataset min and max normalized loss. Min: " +
                "{:.2e}".format(norm_eval_losses.min()) +
                ", Max: " +
                "{:.2e}".format(norm_eval_losses.max()))

            percent_difference = (
                (norm_eval_losses.sum()/eval_sample_number) /
                (norm_test_losses.sum()/test_sample_number)
                ) * 100 - 100
            print(
                "Total Loss Percent diff (eval/test) (%): ",
                percent_difference)

            random_data = test_data[
                random.randint(0, len(test_data))]
            random_data_hat, _ = ae(
                torch.unsqueeze(random_data[0], 0).to(device))

            random_data_image = np.uint8(
                random_data[0].cpu().numpy()*255
                ).transpose(1, 2, 0)
            random_data_hat_image = np.uint8(
                random_data_hat[0].cpu().numpy()*255
                ).transpose(1, 2, 0)
            im_orig = Image.fromarray(random_data_image)
            im_generated = Image.fromarray(random_data_hat_image)

            # im_orig.show("im_orig.png")
            # im_generated.show("im_generated.png")

            if not os.path.exists("save/results"):
                os.makedirs("save/results")

            im_orig.save(
                "save/results/im_orig_" +
                filename.replace(".json", "") + ".png")
            im_generated.save(
                "save/results/im_generated_" +
                filename.replace(".json", "") + ".png")

            random_eval_data = eval_class_data[
                random.randint(0, len(eval_class_data))]
            random_eval_data_hat, _ = ae(
                torch.unsqueeze(random_eval_data[0], 0).to(device))

            random_eval_data_image = np.uint8(
                random_eval_data[0].cpu().numpy()*255
                ).transpose(1, 2, 0)
            random_eval_data_hat_image = np.uint8(
                random_eval_data_hat[0].cpu().numpy()*255
                ).transpose(1, 2, 0)
            im_eval_orig = Image.fromarray(random_eval_data_image)
            im_eval_generated = Image.fromarray(random_eval_data_hat_image)

            im_eval_orig.save(
                "save/results/im_eval_orig_" +
                filename.replace(".json", "") + ".png")
            im_eval_generated.save(
                "save/results/im_eval_generated_" +
                filename.replace(".json", "") + ".png")
            print("")


def scatterDatasetLatent(latentArray):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for latent in tqdm(latentArray):
        ax.scatter(latent[0], latent[1], latent[2])

    ltarr = np.array(latentArray)
    ax.set_xlim3d(ltarr[:, 0].min(), ltarr[:, 0].max())
    ax.set_ylim3d(ltarr[:, 1].min(), ltarr[:, 1].max())
    ax.set_zlim3d(ltarr[:, 2].min(), ltarr[:, 2].max())

    plt.savefig('latent.png')
    # im_scatter = Image.frombytes(
    #     'RGB', fig.canvas.get_width_height(), fig.canvas.tostring_rgb())

    # im_scatter.save("latent.png")


if __name__ == "__main__":
    test_minigrid_memory_binary_ae(random_visualize=False)
    # test_minigrid_memory_ae(random_visualize=False)

    # from visual_pomdp_smm.minigrid_params import params_list
    # for params in params_list:
    #     test_minigrid_memory_ae(params, random_visualize=True)
    #     # test_minigrid_ae(params, random_visualize=True)
    #     # test_minigrid_vae(params, random_visualize=True)
