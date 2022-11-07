from visual_pomdp_smm.testing.minigrid_test import tbLogToPandas
import matplotlib.pyplot as plt
import os

prefix_names = [
        'minigrid_uniform_dynamicobs_ae',
        'minigrid_uniform_dynamicobs_vae',
        'minigrid_uniform_dynamicobs_conv_ae',
        'minigrid_uniform_dynamicobs_conv_vae'
]
filenames = [
        'minigrid_uniform_dynamicobs_ae_2022_11_01_03_13_29_8216',
        'minigrid_uniform_dynamicobs_vae_2022_11_01_03_13_33_6781',
        'minigrid_uniform_dynamicobs_conv_ae_2022_11_01_03_13_33_7016',
        'minigrid_uniform_dynamicobs_conv_vae_2022_11_01_03_13_29_8074'
]

plt.rcParams.update({'font.size': 6})
fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))
fig2, ax2 = plt.subplots(nrows=1, ncols=1, figsize=(10, 6))

for i in range(len(prefix_names)):
    prefix_name = prefix_names[i]
    filename = filenames[i]

    path = (
        "logs/" + prefix_name + "/" +
        filename.replace(prefix_name+"_", "").replace(".json", ""))
    df = tbLogToPandas(path)

    xlabel = "Epoch"
    ylabel = "Loss"
    title = "Average Test Loss Per Epoch "

    test_df = df.loc[df['metric'] == "AvgLossPerEpoch/test"]["value"].rename(
        prefix_name)
    train_df = df.loc[df['metric'] == "AvgLossPerEpoch/train"]["value"].rename(
        prefix_name)

    test_df.plot(
        ax=ax1, legend=True, title=title,
        xlabel=xlabel, ylabel=ylabel)
    train_df.plot(
        ax=ax2, legend=True, title=title,
        xlabel=xlabel, ylabel=ylabel)

fig1.tight_layout(pad=0.3)
fig1.savefig(
    'save/Experiment_Figure_Test_' +
    os.path.commonprefix(prefix_names)+'.png')

fig2.tight_layout(pad=0.3)
fig2.savefig(
    'save/Experiment_Figure_Train_' +
    os.path.commonprefix(prefix_names)+'.png')
# plt.show()
