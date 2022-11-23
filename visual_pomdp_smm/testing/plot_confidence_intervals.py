import os
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import cm
from tensorboard.backend.event_processing import event_accumulator
from tensorboard.backend.event_processing.\
    event_accumulator import EventAccumulator
import natsort
# import pandas as pd

"""
Confidence interval plotting from Tensorboard log data.

Log directories must be in a pre defined form where method name is
seperated with a '-' symbol.

Example

M methods, each is experimented N independent times.

--Log Directory
    --minigrid_memory_conv_AE_16_0
        --2022_11_11_01_54_03_5244
            --events.out.tfevents.1668131643.omtam02.95998.1FC_NN_Last
    --minigrid_memory_conv_AE_16_1
        --2022_11_11_01_54_03_5245
            --events.out.tfevents.1668131644.omtam02.95998.1FC_NN_Last
    --minigrid_memory_conv_AE_16_3
        --2022_11_11_01_54_03_5246
            --events.out.tfevents.1668131645.omtam02.95998.1FC_NN_Last
    --minigrid_memory_conv_binary_AE_256_0
        --2022_11_11_01_54_03_5247
            --events.out.tfevents.1668131647.omtam02.95998.1FC_NN_Last

In this case, this method creates lists of methods
(where method names are lstm, no_memory and oa_k in the above example)
and this script calculates confidence intervals of same methods.

In the confidence interval calculation, since the array sizes of each
run might differ, so all of the arrays are extended with NaNs.
NaN values are not taken into consideration in calculation of
confidence interval at that time step.
Therefore only existing values does not affect the
confidence interval calculation.
"""


def getSRperEpisodeFromDirectory(dpath, scalar_name="AvgLossPerEpoch/test"):
    summary_iterators = [
        EventAccumulator(
            os.path.join(
                dpath,
                dname,
                os.listdir(os.path.join(dpath, dname))[0]),
            size_guidance={
                event_accumulator.SCALARS: 0,
            }).Reload() for dname in os.listdir(dpath)]

    keys = list(set([
        summary.path.replace(dpath, '').rsplit('/', 1)[0].rsplit('_', 1)[0]
        for summary in summary_iterators]))
    keys = sorted(keys)
    indexes = {k: [] for k in keys}
    values = {k: [] for k in keys}

    for summary in summary_iterators:
        experiment_name = summary.path.replace(dpath, '').\
            rsplit('/', 1)[0].rsplit('_', 1)[0]
        summary_scalar = summary.Scalars(scalar_name)
        indexes[experiment_name].append(np.fromiter
                                        (map(lambda x: x.step, summary_scalar),
                                         dtype=int))
        values[experiment_name].append(np.fromiter(
            map(lambda x: x.value, summary_scalar), dtype=np.double))

    return indexes, values, keys


def savePlotSRWithCI(indexes, values, keys, save_name, legend_prefix):
    plt.rcParams.update({'font.size': 18})
    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_title("Average Test Loss Per Epoch")
    # color = iter(cm.brg(np.linspace(0, 1, len(keys))))
    color = iter(cm.jet(np.linspace(0, 1, len(keys))))
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Loss")
    lines = []
    # keys = sorted(keys)
    keys = natsort.natsorted(keys, reverse=False)
    legend_keys = [
        legend_prefix + key.rsplit('_', 1)[1] for key in keys]
    for key in keys:
        indexesList = indexes[key]
        valuesList = values[key]
        numberOfExperiments = len(indexesList)
        maxLength = np.max(np.array
                           ([len(indexList) for indexList in indexesList]))
        x = np.arange(1, maxLength+1, 1)
        extendList = valuesList
        for i in range(len(valuesList)):
            extend = np.empty((maxLength,))
            extend[:] = np.nan
            extend[:valuesList[i].shape[0]] = valuesList[i]
            extendList[i] = extend

        y = np.stack(extendList, axis=0)
        # %95 confidence interval.
        ci = 1.96 * np.nanstd(y, axis=0)/np.sqrt((numberOfExperiments))
        mean_y = np.nanmean(y, axis=0)

        c = next(color)
        line, = ax.plot(x, mean_y, color=c, zorder=100)
        lines.append(line)
        ax.fill_between(
            x, (mean_y-ci), (mean_y+ci), color=c, alpha=.2, zorder=0)

    # ax.set_ylim(0, 0.2)
    ax.legend(lines, legend_keys)
    # fig.savefig(save_name, format="png", bbox_inches="tight")
    fig.savefig(save_name, format="png")


def MemoryLatentComparison():
    path = 'save/csv/memory_conv_ae/'
    save_name = 'results/LatentComparison_Memory_Conv_AE.png'

    indexes, values, keys = getSRperEpisodeFromDirectory(path)
    print("Read from file has been completed:", path)
    savePlotSRWithCI(
        indexes, values, keys, save_name,
        legend_prefix="Latent Dimension Size=")
    print(save_name, " has been generated.")

    path = 'save/csv/memory_conv_binary/'
    save_name = 'results/LatentComparison_Memory_Conv_Binary_AE.png'

    indexes, values, keys = getSRperEpisodeFromDirectory(path)
    print("Read from file has been completed:", path)
    savePlotSRWithCI(indexes, values, keys, save_name)
    print(save_name, " has been generated.")

    print("Completed all plots.")


def DynamicObsLatentComparison():
    path = 'save/csv/dynamicobs_conv_ae/'
    save_name = 'results/LatentComparison_DynamicObs_Conv_AE.png'

    indexes, values, keys = getSRperEpisodeFromDirectory(path)
    print("Read from file has been completed:", path)
    savePlotSRWithCI(
        indexes, values, keys, save_name,
        legend_prefix="Latent Dimension Size=")
    print(save_name, " has been generated.")

    path = 'save/csv/dynamicobs_conv_binary/'
    save_name = 'results/LatentComparison_DynamicObs_Conv_Binary_AE.png'

    indexes, values, keys = getSRperEpisodeFromDirectory(path)
    print("Read from file has been completed:", path)
    savePlotSRWithCI(indexes, values, keys, save_name)
    print(save_name, " has been generated.")

    print("Completed all plots.")


def UniformMemoryCompareTraining():
    path = 'logs/'
    save_name = 'results/UniformMemoryTrainingComparison.png'

    indexes, values, keys = getSRperEpisodeFromDirectory(path)
    print("Read from file has been completed:", path)
    savePlotSRWithCI(indexes, values, keys, save_name, legend_prefix="Method=")
    print(save_name, " has been generated.")

    print("Completed all plots.")


def UniformDynamicObsCompareTraining():
    path = 'logs/'
    save_name = 'results/UniformDynamicObsTrainingComparison.png'

    indexes, values, keys = getSRperEpisodeFromDirectory(path)
    print("Read from file has been completed:", path)
    savePlotSRWithCI(indexes, values, keys, save_name, legend_prefix="Method=")
    print(save_name, " has been generated.")

    print("Completed all plots.")


def SequenceMemoryCompareTraining():
    path = 'logs/'
    save_name = 'results/SequenceMemoryTrainingComparison.png'

    indexes, values, keys = getSRperEpisodeFromDirectory(path)
    print("Read from file has been completed:", path)
    savePlotSRWithCI(indexes, values, keys, save_name, legend_prefix="Method=")
    print(save_name, " has been generated.")

    print("Completed all plots.")


def SequenceDynamicObsCompareTraining():
    path = 'logs/'
    save_name = 'results/SequenceDynamicObsTrainingComparison.png'

    indexes, values, keys = getSRperEpisodeFromDirectory(path)
    print("Read from file has been completed:", path)
    savePlotSRWithCI(indexes, values, keys, save_name, legend_prefix="Method=")
    print(save_name, " has been generated.")

    print("Completed all plots.")


def plotFreqVsReconsLossWithCI(filename, legend_prefix):

    freq_vs_losses_dict_main = np.load(filename, allow_pickle=True).item()

    keys = list(freq_vs_losses_dict_main.keys())
    unique_keys = set([key.rsplit('_', 1)[0] for key in keys])
    unique_keys = natsort.natsorted(unique_keys, reverse=False)

    color = iter(cm.jet(np.linspace(0, 1, len(unique_keys))))
    lines1 = []
    lines2 = []
    legend_keys = [
        legend_prefix + key for key in unique_keys]
    legend_keys.append("Visitation Count")

    fig1, ax1 = plt.subplots(figsize=(16, 9))
    fig2, ax2 = plt.subplots(figsize=(16, 9))
    ax1_twin = ax1.twinx()
    ax2_twin = ax2.twinx()

    color = iter(cm.jet(np.linspace(0, 1, len(unique_keys))))

    for unique_key in unique_keys:
        key_dict = {
            k: v for k, v in freq_vs_losses_dict_main.items()
            if k.startswith(unique_key)}
        numberOfExperiments = len(key_dict)
        sorted_freq = np.array([
            v['sorted_frequencies']
            for k, v in key_dict.items()])
        normalized_top_n_losses_array = np.array([
            v['normalized_top_n_losses_array']
            for k, v in key_dict.items()])
        normalized_top_n_losses_array_clip = np.array([
            v['normalized_top_n_losses_array_clip']
            for k, v in key_dict.items()])

        mean_sorted_freq = np.nanmean(
            sorted_freq, axis=0)
        mean_sorted_freq = mean_sorted_freq / mean_sorted_freq.sum()
        index_values = np.arange(0, len(mean_sorted_freq), 1)
        # ci_sorted_freq = 1.96 * (
        #     np.nanstd(sorted_freq, axis=0) /
        #     np.sqrt((numberOfExperiments)))

        mean_normalized_top_n_losses_array = np.nanmean(
            normalized_top_n_losses_array, axis=0)
        ci_normalized_top_n_losses_array = 1.96 * (
            np.nanstd(normalized_top_n_losses_array, axis=0) /
            np.sqrt((numberOfExperiments)))

        mean_normalized_top_n_losses_array_clip = np.nanmean(
            normalized_top_n_losses_array_clip, axis=0)
        ci_normalized_top_n_losses_array_clip = 1.96 * (
            np.nanstd(normalized_top_n_losses_array_clip, axis=0) /
            np.sqrt((numberOfExperiments)))

        c = next(color)

        line1_, = ax1.plot(
            index_values, mean_sorted_freq, color='g',
            zorder=100, linewidth=4)
        line1, = ax1_twin.plot(
            index_values,
            mean_normalized_top_n_losses_array,
            color=c, zorder=100)

        lines1.append(line1)
        ax1_twin.fill_between(
            index_values,
            (mean_normalized_top_n_losses_array -
                ci_normalized_top_n_losses_array),
            (mean_normalized_top_n_losses_array +
                ci_normalized_top_n_losses_array),
            color=c, alpha=.2, zorder=0)

        line2_, = ax2.plot(
            index_values, mean_sorted_freq, color='g',
            zorder=100, linewidth=4)
        line2, = ax2_twin.plot(
            index_values,
            mean_normalized_top_n_losses_array_clip,
            color=c, zorder=100)
        lines2.append(line2)
        ax2_twin.fill_between(
            index_values,
            (mean_normalized_top_n_losses_array_clip -
                ci_normalized_top_n_losses_array_clip),
            (mean_normalized_top_n_losses_array_clip +
                ci_normalized_top_n_losses_array_clip),
            color=c, alpha=.2, zorder=0)

    ax1.set_title(
        "Visitation Count Versus Autoencoder Reconstruction Error",
        fontsize=18)
    ax2.set_title(
        "Visitation Count Versus Autoencoder Reconstruction Error",
        fontsize=18)

    lines1.append(line1_)
    lines2.append(line2_)
    ax1_twin.legend(lines1, legend_keys, loc='upper right').set_zorder(200)
    ax2_twin.legend(lines2, legend_keys, loc='upper right').set_zorder(200)
    ax1.set_xticks([])
    ax1.set_xlabel('Observations', fontsize=18)
    ax1.set_ylabel(
        'Normalized Visit Count / Frequency', color='g', fontsize=18)
    ax1_twin.set_ylabel('Reconstruction Error', color='k', fontsize=18)

    ax2.set_xticks([])
    ax2.set_xlabel('Observations', fontsize=18)
    ax2.set_ylabel(
        'Normalized Visit Count / Frequency', color='g', fontsize=18)
    ax2_twin.set_ylabel('Reconstruction Error', color='k',  fontsize=18)

    fig1.savefig(
        filename.replace("Dict.npy", "") + "NoClipping.png", format="png")
    fig2.savefig(
        filename.replace("Dict.npy", "") + "WithClipping.png", format="png")


if __name__ == "__main__":
    # MemoryLatentComparison()
    # DynamicObsLatentComparison()
    # UniformMemoryCompareTraining()
    # UniformDynamicObsCompareTraining()
    # SequenceMemoryCompareTraining()
    # SequenceDynamicObsCompareTraining()

    plotFreqVsReconsLossWithCI(
        filename=(
            "save/" +
            "Experiment_Test_Uniform_Memory_Freq_Vs_Losses_Dict.npy"),
        legend_prefix="")

    # plotFreqVsReconsLossWithCI(
    #     filename=(
    #         "save/" +
    #         "Experiment_Test_Uniform_Dynamic_Obs_Freq_Vs_Losses_Dict.npy"),
    #     legend_prefix="")

    # plotFreqVsReconsLossWithCI(
    #     filename=(
    #         "save/" +
    #         "Experiment_Test_Sequence_Memory_Freq_Vs_Losses_Dict.npy"),
    #     legend_prefix="")

    # plotFreqVsReconsLossWithCI(
    #     filename=(
    #         "save/" +
    #         "Experiment_Test_Sequence_Dynamic_Obs_Freq_Vs_Losses_Dict.npy"),
    #     legend_prefix="")
