import torch
import matplotlib.pyplot as plt
import os
from PIL import Image
import random
import numpy as np
from visual_pomdp_smm.minigrid_utils import MinigridDataset
from visual_pomdp_smm.minigrid_utils import batch_size, train_set_ratio,\
    input_dims

plt.rcParams['figure.dpi'] = 200
torch.manual_seed(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def test_model(
        autoencoder, test_dataset):

    total_test_loss = 0
    autoencoder.eval()
    with torch.no_grad():
        # for x, y in test_dataset:
        for batch_idx, (x, y) in enumerate(test_dataset):
            x = x.to(device)
            x_hat = autoencoder(x)
            loss = ((x - x_hat)**2).sum()
            total_test_loss += loss.item()

    return total_test_loss


def test_minigrid_ae(random_visualize=False):
    prefix_name = "minigrid_VAE_"
    dirFiles = os.listdir('save')
    prefixed = [filename for filename in dirFiles
                if filename.startswith(prefix_name)]

    prefixed.sort(reverse=True)
    model_file_name = prefixed[0]

    ae = torch.load("save/" + model_file_name).to(device)
    ae.eval()

    test_data = MinigridDataset(
        "data/", "test",
        image_size=input_dims, train_set_ratio=train_set_ratio)

    test_dataset = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=True,
        num_workers=1, pin_memory=True)

    test_loss = test_model(ae, test_dataset)
    print(test_loss)


def test_minigrid_vae(random_visualize=False):
    prefix_name = "minigrid_VAE_"
    dirFiles = os.listdir('save')
    prefixed = [filename for filename in dirFiles
                if filename.startswith(prefix_name)]

    prefixed.sort(reverse=True)
    model_file_name = prefixed[0]
    with torch.no_grad():

        vae = torch.load("save/" + model_file_name).to(device)
        vae.eval()

        test_data = MinigridDataset(
            "data/", "test",
            image_size=input_dims, train_set_ratio=train_set_ratio,
            use_cache=False)

        test_dataset = torch.utils.data.DataLoader(
            test_data, batch_size=batch_size, shuffle=True,
            num_workers=1, pin_memory=True)

        test_loss = test_model(vae, test_dataset)
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
    pass


if __name__ == "__main__":
    # test_minigrid_ae(random_visualize=True)
    test_minigrid_vae(random_visualize=True)
