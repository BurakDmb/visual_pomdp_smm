import torch
# import matplotlib
import matplotlib.pyplot as plt
import os
from PIL import Image
import random
import numpy as np
from tqdm.auto import tqdm
from visual_pomdp_smm.minigrid_utils import MinigridDataset
from visual_pomdp_smm.minigrid_utils import batch_size, train_set_ratio,\
    input_dims

# plt.rcParams['figure.dpi'] = 200
# matplotlib.use('GTK3Agg')

torch.manual_seed(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def test_model(
        autoencoder, test_dataset):

    total_test_loss = 0
    latentArray = []
    autoencoder.eval()
    with torch.no_grad():
        # for x, y in test_dataset:
        for batch_idx, (x, y) in enumerate(test_dataset):
            x = x.to(device)
            x_hat = autoencoder(x)
            loss = ((x - x_hat)**2).sum()
            total_test_loss += loss.item()
            latentArray.extend(autoencoder.z.cpu().numpy().tolist())

    return total_test_loss, latentArray


def test_minigrid_ae(random_visualize=False):
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
            image_size=input_dims, train_set_ratio=train_set_ratio)

        test_dataset = torch.utils.data.DataLoader(
            test_data, batch_size=128, shuffle=True,
            num_workers=1, pin_memory=True)

        test_loss, latentArray = test_model(ae, test_dataset)
        print(test_loss)

        random_data = test_data[
            random.randint(0, len(test_data))]
        random_data_hat = ae(torch.unsqueeze(random_data[0], 0).to(device))

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
        im_orig.save("im_orig.png")
        im_generated.save("im_generated.png")
        scatterDatasetLatent(latentArray)


def test_minigrid_vae(random_visualize=False):
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
            image_size=input_dims, train_set_ratio=train_set_ratio,
            use_cache=False)

        test_dataset = torch.utils.data.DataLoader(
            test_data, batch_size=batch_size, shuffle=True,
            num_workers=1, pin_memory=True)

        test_loss, latentArray = test_model(vae, test_dataset)
        print(test_loss)
        random_data = test_data[
            random.randint(0, len(test_data))]
        random_data_hat = vae(torch.unsqueeze(random_data[0], 0).to(device))

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
    test_minigrid_ae(random_visualize=True)
    # test_minigrid_vae(random_visualize=True)
