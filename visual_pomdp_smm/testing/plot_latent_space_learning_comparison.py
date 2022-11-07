import pandas as pd
import matplotlib.pyplot as plt

plot_binary = False
if not plot_binary:

    ae_lat16 = pd.read_csv(
        'results/csv/ae_latent16.csv')['Value'].rename("Latent_dim=16")
    ae_lat32 = pd.read_csv(
        'results/csv/ae_latent32.csv')['Value'].rename("Latent_dim=32")
    ae_lat64 = pd.read_csv(
        'results/csv/ae_latent64.csv')['Value'].rename("Latent_dim=64")
    ae_lat128 = pd.read_csv(
        'results/csv/ae_latent128.csv')['Value'].rename("Latent_dim=128")
    ae_lat256 = pd.read_csv(
        'results/csv/ae_latent256.csv')['Value'].rename("Latent_dim=256")

    xlabel = "Epoch"
    ylabel = "Loss"
    title = "Average Test Loss Per Epoch "

    ae_lat16.plot(
        ylim=(0, 400), legend=True, title=title, xlabel=xlabel, ylabel=ylabel)
    ae_lat32.plot(
        ylim=(0, 400), legend=True, title=title, xlabel=xlabel, ylabel=ylabel)
    ae_lat64.plot(
        ylim=(0, 400), legend=True, title=title, xlabel=xlabel, ylabel=ylabel)
    ae_lat128.plot(
        ylim=(0, 400), legend=True, title=title, xlabel=xlabel, ylabel=ylabel)
    ae_lat256.plot(
        ylim=(0, 400), legend=True, title=title, xlabel=xlabel, ylabel=ylabel)
    # plt.show()
    plt.savefig("results/AE_AvgTestLoss.png")
else:
    ae_b_lat16 = pd.read_csv(
        'results/csv/ae_b_latent16.csv')['Value'].rename("Latent_dim=16")
    ae_b_lat32 = pd.read_csv(
        'results/csv/ae_b_latent32.csv')['Value'].rename("Latent_dim=32")
    ae_b_lat64 = pd.read_csv(
        'results/csv/ae_b_latent64.csv')['Value'].rename("Latent_dim=64")
    ae_b_lat128 = pd.read_csv(
        'results/csv/ae_b_latent128.csv')['Value'].rename("Latent_dim=128")
    ae_b_lat256 = pd.read_csv(
        'results/csv/ae_b_latent256.csv')['Value'].rename("Latent_dim=256")

    xlabel = "Epoch"
    ylabel = "Loss"
    title = "Average Test Loss Per Epoch "

    ae_b_lat16.plot(
        ylim=(0, 0.125), legend=True,
        title=title, xlabel=xlabel, ylabel=ylabel)
    ae_b_lat32.plot(
        ylim=(0, 0.125), legend=True,
        title=title, xlabel=xlabel, ylabel=ylabel)
    ae_b_lat64.plot(
        ylim=(0, 0.125), legend=True,
        title=title, xlabel=xlabel, ylabel=ylabel)
    ae_b_lat128.plot(
        ylim=(0, 0.125), legend=True,
        title=title, xlabel=xlabel, ylabel=ylabel)
    ae_b_lat256.plot(
        ylim=(0, 0.125), legend=True,
        title=title, xlabel=xlabel, ylabel=ylabel)
    # plt.show()
    plt.savefig("results/Binary_AE_AvgTestLoss.png")
