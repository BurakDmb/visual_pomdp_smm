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


def savePlotSRWithCI(indexes, values, keys, save_name):
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
        "Latent Dimension Size=" + key.rsplit('_', 1)[1] for key in keys]
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


def main():
    path = 'save/csv/memory_conv_ae/'
    save_name = 'results/LatentComparison_Memory_Conv_AE.png'

    indexes, values, keys = getSRperEpisodeFromDirectory(path)
    print("Read from file has been completed:", path)
    savePlotSRWithCI(indexes, values, keys, save_name)
    print(save_name, " has been generated.")

    path = 'save/csv/memory_conv_binary/'
    save_name = 'results/LatentComparison_Memory_Conv_Binary_AE.png'

    indexes, values, keys = getSRperEpisodeFromDirectory(path)
    print("Read from file has been completed:", path)
    savePlotSRWithCI(indexes, values, keys, save_name)
    print(save_name, " has been generated.")

    print("Completed all plots.")
    pass


if __name__ == "__main__":
    main()
