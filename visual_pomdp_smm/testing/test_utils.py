import json
import os
import random

# import matplotlib
import matplotlib.pyplot as plt
import natsort
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from tensorboard.backend.event_processing.event_accumulator import \
    EventAccumulator
from tqdm.auto import tqdm

from visual_pomdp_smm.envs.minigrid.minigrid_utils import (
    MinigridGenericDataset, MinigridGenericDatasetEval,
    MinigridGenericDatasetNoteval)

# plt.rcParams['figure.dpi'] = 200
# matplotlib.use('GTK3Agg')

# torch.manual_seed(0)
# random.seed(None)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def test_model(
        autoencoder, test_dataset):

    test_losses = []
    total_test_loss = 0
    total_sample_number = 0
    latentArray = []
    lossesArray = []
    autoencoder.eval()

    loss_func = nn.L1Loss(reduction="none")
    # loss_func = nn.BCELoss(reduction='none')

    with torch.no_grad():
        # for x, y in test_dataset:
        for batch_idx, (x, y) in enumerate(test_dataset):
            total_sample_number += len(x)
            x = x.to(device)
            x_hat, z = autoencoder(x)
            loss_values = loss_func(x_hat, x)
            loss = loss_values.sum()
            losses = loss_values.sum((1, 2, 3)).cpu().numpy().tolist()
            lossesArray.extend(losses)
            total_test_loss += loss.item()
            test_losses.append(loss.item())
            latentArray.extend(z.cpu().numpy().tolist())

    return total_test_loss, total_sample_number,\
        latentArray, test_losses, lossesArray


def test_function(
        prefix_name_inputs, random_visualize=False, save_figures=False,
        verbose=False, include_all_experiments=False, n_frequency=500):

    dirFiles = os.listdir('save/json')
    if type(prefix_name_inputs) is not list:
        prefixes = [prefix_name_inputs]
    else:
        prefixes = prefix_name_inputs

    resultsDictMain = {}
    freq_vs_losses_dict_main = {}
    prev_image_size = None
    prev_train_set_ratio = None
    fig1, ax1 = plt.subplots(nrows=1, ncols=1)
    fig2, ax2 = plt.subplots(nrows=1, ncols=1)
    for prefix_name in tqdm(prefixes):

        prefixed = [filename for filename in dirFiles
                    if filename.startswith(prefix_name)]

        prefixed.sort(reverse=True)
        if not include_all_experiments:
            prefixed = [prefixed[0]]
            current_full_prefix = [prefix_name]
            # TODO: Not completed, after_prefix missing.
            # filename.replace(prefix_name+"_", "")
        else:
            current_full_prefix = [
                prefix_name + "_" +
                prefix.split(prefix_name+"_", 1)[1].split('_', 1)[0]
                for prefix in prefixed]

            after_prefix = [
                prefix.split(prefix_name+"_", 1)[1].split('_', 1)[1]
                for prefix in prefixed]

        for i, filename in enumerate(prefixed):
            with open("save/json/"+filename, 'r') as params_file:
                params = json.loads(params_file.read())
            model_file_name = params['save_path']
            print("Testing for param path: " + params['save_path'])

            resultsDict = {}
            freq_vs_losses_dict = {}

            with torch.no_grad():
                ae = torch.load(
                    model_file_name+".torch", map_location='cuda:0').to(device)
                if hasattr(ae, 'module'):
                    ae = ae.module
                ae.eval()

                if "input_dims_h" not in params:
                    params["input_dims_h"] = params["input_dims"]
                    params["input_dims_w"] = params["input_dims"]

                if not (prev_image_size is not None
                        and prev_image_size == params['input_dims_h']
                        and prev_train_set_ratio is not None
                        and prev_train_set_ratio == params['train_set_ratio']):
                    print("Creating dataset")

                    test_data = MinigridGenericDatasetNoteval(
                        "data/", "test",
                        image_size_h=params['input_dims_h'],
                        image_size_w=params['input_dims_w'],
                        train_set_ratio=params['train_set_ratio'],
                        dataset_folder_name=params['dataset_folder_name'],
                        use_cache=True)

                    test_dataset = torch.utils.data.DataLoader(
                        test_data, batch_size=16800, shuffle=False,
                        num_workers=0, pin_memory=False)

                    eval_class_data = MinigridGenericDatasetEval(
                        "data/", "",
                        image_size_h=params['input_dims_h'],
                        image_size_w=params['input_dims_w'],
                        train_set_ratio=params['train_set_ratio'],
                        dataset_folder_name=params['dataset_folder_name'],
                        use_cache=True)

                    eval_class_dataset = torch.utils.data.DataLoader(
                        eval_class_data, batch_size=16800, shuffle=False,
                        num_workers=0, pin_memory=False)

                    random_data = test_data[
                        random.randint(0, len(test_data))]
                    random_eval_data = eval_class_data[
                        random.randint(0, len(eval_class_data))]

                    all_data = MinigridGenericDataset(
                        "data/", "test",
                        image_size_h=params['input_dims_h'],
                        image_size_w=params['input_dims_w'],
                        train_set_ratio=params['train_set_ratio'],
                        dataset_folder_name=params['dataset_folder_name'],
                        use_cache=True)

                    unique_values, indices, counts = np.unique(
                        all_data.imgs, return_counts=True,
                        return_index=True, axis=0)
                    zipped_array = zip(counts, indices)
                    sorted_zip_frequencies = sorted(
                        zipped_array, reverse=True)[0:n_frequency]

                    sorted_frequencies, sorted_indices = zip(
                        *sorted_zip_frequencies)
                    all_data.imgs = all_data.imgs[np.array(sorted_indices)]
                    all_dataset = torch.utils.data.DataLoader(
                        all_data, batch_size=len(sorted_frequencies),
                        shuffle=False,
                        num_workers=0, pin_memory=False)

                    print("Finished creating dataset.")

                # Calculating all dataset results.
                top_n_eval_loss, top_n_sample_number, top_n_LatentArrays,\
                    top_n_losses, top_n_LossesArray = test_model(
                        ae, all_dataset)

                normalized_top_n_losses_array = (
                    np.array(top_n_LossesArray) / (
                        params['input_dims_h'] *
                        params['input_dims_w'] *
                        params['in_channels'])
                    )
                # We should clip the values if its exceeds
                # 10x larger than the minimum loss
                normalized_top_n_losses_array_clip = np.clip(
                    normalized_top_n_losses_array,
                    normalized_top_n_losses_array.min(),
                    normalized_top_n_losses_array.min()*10)
                # freq_index_loss_zip = zip(
                #     sorted_frequencies,
                #     sorted_indices,
                #     normalized_top_n_losses_array)
                freq_vs_losses_dict['sorted_frequencies'] = sorted_frequencies
                freq_vs_losses_dict['sorted_indices'] = sorted_indices
                freq_vs_losses_dict['normalized_top_n_losses_array'] = \
                    normalized_top_n_losses_array
                freq_vs_losses_dict['normalized_top_n_losses_array_clip'] = \
                    normalized_top_n_losses_array_clip

                # Calculating noteval dataset results.
                total_test_loss, test_sample_number, latentArrays,\
                    test_losses, testLossesArray = test_model(ae, test_dataset)

                resultsDict['filename'] = filename

                norm_test_losses = (
                    np.array(testLossesArray) / (
                        params['input_dims_h'] *
                        params['input_dims_w'] *
                        params['in_channels'])
                    )
                resultsDict['test_avgloss'] = norm_test_losses.sum(
                    )/test_sample_number
                resultsDict['test_minloss'] = norm_test_losses.min()
                resultsDict['test_maxloss'] = norm_test_losses.max()

                # Calculating eval dataset results.
                total_eval_loss, eval_sample_number, evalLatentArrays,\
                    eval_losses, evalLossesArray = test_model(
                        ae, eval_class_dataset)
                norm_eval_losses = (
                    np.array(evalLossesArray) / (
                        params['input_dims_h'] *
                        params['input_dims_w'] *
                        params['in_channels'])
                    )

                resultsDict['eval_avgloss'] = norm_eval_losses.sum(
                    )/eval_sample_number
                resultsDict['eval_minloss'] = norm_eval_losses.min()
                resultsDict['eval_maxloss'] = norm_eval_losses.max()

                percent_difference = (
                    (norm_eval_losses.sum()/eval_sample_number) /
                    (norm_test_losses.sum()/test_sample_number)
                    ) * 100 - 100
                resultsDict['lossdiff'] = percent_difference

                if verbose:
                    print("Filename: ", filename)
                    print(
                        "Test Dataset AvgLoss: ",
                        "{:.2e}".format(resultsDict['test_avgloss']))
                    print(
                        "Test Dataset min and max normalized loss. Min: " +
                        "{:.2e}".format(resultsDict['test_minloss']) +
                        ", Max: " +
                        "{:.2e}".format(resultsDict['test_maxloss']))

                    print(
                        "Eval Class Dataset AvgLoss",
                        "{:.2e}".
                        format(norm_eval_losses.sum()/eval_sample_number))
                    print(
                        "Eval Class Dataset min and max normalized loss. Min: "
                        +
                        "{:.2e}".format(norm_eval_losses.min()) +
                        ", Max: " +
                        "{:.2e}".format(norm_eval_losses.max()))
                    print(
                        "Total Loss Percent diff (eval/test) (%): ",
                        percent_difference)

                if save_figures:

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

                    if not os.path.exists("save/figures"):
                        os.makedirs("save/figures")

                    im_orig.save(
                        "save/figures/" + prefix_name +
                        filename.replace(prefix_name+"_", "").
                        replace(".json", "") +
                        "_im_orig" + ".png")
                    im_generated.save(
                        "save/figures/" + prefix_name +
                        filename.replace(prefix_name+"_", "")
                        .replace(".json", "") +
                        "_im_generated" + ".png")

                    random_eval_data_hat, _ = ae(
                        torch.unsqueeze(random_eval_data[0], 0).to(device))

                    random_eval_data_image = np.uint8(
                        random_eval_data[0].cpu().numpy()*255
                        ).transpose(1, 2, 0)
                    random_eval_data_hat_image = np.uint8(
                        random_eval_data_hat[0].cpu().numpy()*255
                        ).transpose(1, 2, 0)
                    im_eval_orig = Image.fromarray(
                        random_eval_data_image)
                    im_eval_generated = Image.fromarray(
                        random_eval_data_hat_image)

                    im_eval_orig.save(
                        "save/figures/"+prefix_name +
                        filename.replace(prefix_name+"_", "")
                        .replace(".json", "") +
                        "_im_eval" + ".png")
                    im_eval_generated.save(
                        "save/figures/"+prefix_name +
                        filename.replace(prefix_name+"_", "")
                        .replace(".json", "") +
                        "_im_eval_generated" + ".png")

                    # path = (
                    #     "logs/" + prefix_name + "/" +
                    #     filename.replace(prefix_name+"_", "").
                    #     replace(".json", ""))

                    df = tbLogToPandas(
                        "logs/"+current_full_prefix[i] + "/" +
                        after_prefix[i].replace(".json", ""))

                    xlabel = "Epoch"
                    ylabel = "Loss"
                    title = "Average Test Loss Per Epoch"

                    test_df = df.loc[
                            df['metric'] == "AvgLossPerEpoch/test"
                        ]["value"].rename(
                            prefix_name)
                    train_df = df.loc[
                            df['metric'] == "AvgLossPerEpoch/train"
                        ]["value"].rename(
                            prefix_name)

                    test_df.plot(
                        ax=ax1, legend=True, title=title,
                        xlabel=xlabel, ylabel=ylabel)
                    train_df.plot(
                        ax=ax2, legend=True, title=title,
                        xlabel=xlabel, ylabel=ylabel)

            resultsDictMain[current_full_prefix[i]] = resultsDict
            freq_vs_losses_dict_main[current_full_prefix[i]] = \
                freq_vs_losses_dict
            prev_image_size = params['input_dims_h']
            prev_train_set_ratio = params['train_set_ratio']

    if save_figures:
        fig1.tight_layout(pad=0.3)
        fig1.savefig(
            'save/Experiment_Figure_Test_' +
            os.path.commonprefix(prefixes)+'.png')

        fig2.tight_layout(pad=0.3)
        fig2.savefig(
            'save/Experiment_Figure_Train_' +
            os.path.commonprefix(prefixes)+'.png')

        # fig3.tight_layout(pad=0.3)
        # fig3.savefig(
        #     'save/Experiment_Figure_FreqVsLoss' +
        #     os.path.commonprefix(prefixes)+'.png')

        # fig4.tight_layout(pad=0.3)
        # fig4.savefig(
        #     'save/Experiment_Figure_FreqVsLoss_Clipped' +
        #     os.path.commonprefix(prefixes)+'.png')
    return resultsDictMain, freq_vs_losses_dict_main


def _plotfreq(
        filename, sorted_frequencies,
        normalized_top_n_losses_array,
        normalized_top_n_losses_array_clip):
    # freq_vs_losses_dict_main = np.load(filename).item()
    # https://stackoverflow.com/a/40220343/9151122

    fig3, ax3 = plt.subplots(nrows=1, ncols=1, figsize=(16, 9))
    fig4, ax4 = plt.subplots(nrows=1, ncols=1, figsize=(16, 9))
    # Frequency Of Observations Vs Reconstruction Loss Figures.
    # Fig3: Without clipping, Fig4: With Clipping
    index_values = np.arange(0, len(sorted_frequencies), 1)
    ax3_twin = ax3.twinx()
    ax4_twin = ax4.twinx()

    ax3.plot(
        index_values, sorted_frequencies, 'g-')
    ax3_twin.plot(
        index_values,
        normalized_top_n_losses_array, 'b-')

    ax4.plot(
        index_values, sorted_frequencies, 'g-')
    ax4_twin.plot(
        index_values,
        normalized_top_n_losses_array_clip, 'b-')

    ax3.set_xticks([])
    ax3.set_xlabel('Observations')
    ax3.set_ylabel('Frequency', color='g')
    ax3_twin.set_ylabel('Reconstruction Error', color='b')

    ax4.set_xticks([])
    ax4.set_xlabel('Observations')
    ax4.set_ylabel('Frequency', color='g')
    ax4_twin.set_ylabel('Reconstruction Error', color='b')


def calculate_std_table(filename):
    with open(filename, "r") as infile:
        json_dict = json.load(infile)

        keys = list(json_dict.keys())
        unique_keys = set([key.rsplit('_', 1)[0] for key in keys])
        unique_keys = natsort.natsorted(unique_keys, reverse=False)
        results_dict = {}

        for unique_key in unique_keys:
            key_dict = {
                k: v for k, v in json_dict.items() if k.startswith(unique_key)}

            results_dict[unique_key] = {}
            results_dict[unique_key]['test_avgloss_mean'] = np.mean([
                v['test_avgloss'] for k, v in key_dict.items()])
            results_dict[unique_key]['test_avgloss_std'] = np.std([
                v['test_avgloss'] for k, v in key_dict.items()])

            results_dict[unique_key]['eval_avgloss_mean'] = np.mean([
                v['eval_avgloss'] for k, v in key_dict.items()])
            results_dict[unique_key]['eval_avgloss_std'] = np.std([
                v['eval_avgloss'] for k, v in key_dict.items()])

            results_dict[unique_key]['lossdiff_mean'] = np.mean([
                v['lossdiff'] for k, v in key_dict.items()])
            results_dict[unique_key]['lossdiff_std'] = np.std([
                v['lossdiff'] for k, v in key_dict.items()])

        print()
        json_result_dict = json.dumps(results_dict, indent=2, default=str)
        print(json_result_dict)

        if not os.path.exists("save"):
            os.makedirs("save")
        # Writing to sample.json
        with open(
                filename.replace(".json", "")+"_Stochastic_Results.json",
                "w") as outfile:
            outfile.write(json_result_dict)


# Extraction function
def tbLogToPandas(path):
    runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})
    try:
        event_acc = EventAccumulator(path)
        event_acc.Reload()
        tags = event_acc.Tags()["scalars"]
        for tag in tags:
            event_list = event_acc.Scalars(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            r = {"metric": [tag] * len(step), "value": values, "step": step}
            r = pd.DataFrame(r)
            runlog_data = pd.concat([runlog_data, r])
    # Dirty catch of DataLossError
    except Exception as e:
        print("Event file possibly corrupt: {}".format(path))
        print("Exception message:\n", e)
        exit(1)
    return runlog_data


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


# if __name__ == "__main__":
#     test_minigrid_memory_binary_ae(random_visualize=False)
#     # test_minigrid_memory_ae(random_visualize=False)

#     # from visual_pomdp_smm.minigrid_params import params_list
#     # for params in params_list:
#     #     test_minigrid_memory_ae(params, random_visualize=True)
#     #     # test_minigrid_ae(params, random_visualize=True)
#     #     # test_minigrid_vae(params, random_visualize=True)
