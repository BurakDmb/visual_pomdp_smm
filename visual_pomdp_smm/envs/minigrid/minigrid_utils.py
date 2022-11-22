import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributions
# import torch.nn as nn
# import torch.nn.functional as F
import torch.utils
import torchvision

# import matplotlib.pyplot as plt


# plt.rcParams['figure.dpi'] = 200
# torch.manual_seed(0)
# device = 'cuda' if torch.cuda.is_available() else 'cpu'


class MinigridGenericDataset(torch.utils.data.Dataset):
    def __init__(
            self, data_path, split, image_size_h, image_size_w,
            train_set_ratio,
            dataset_folder_name, use_cache=False, **kwargs):

        self.data_dir = Path(data_path) / dataset_folder_name
        if not os.path.isdir(self.data_dir):
            print("Given data directory does not exists: ", self.data_dir)

        self.transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((image_size_h, image_size_w)), ])

        dataset_dict = json.load(open(self.data_dir/'dataset_dict.json', 'r'))
        if not dataset_dict:
            print("Dataset dictionary file is empty, exiting the program.")
            exit(1)

        if split == "eval":
            eval_states_list = np.memmap(
                self.data_dir/'sample_eval.npy',
                dtype='uint8', mode='r',
                shape=(
                    tuple(dataset_dict['eval_states_shape'])))
            self.imgs = eval_states_list

        elif split == "noteval":
            noteval_states_list = np.memmap(
                self.data_dir/'sample_noteval.npy',
                dtype='uint8', mode='r',
                shape=(
                    tuple(dataset_dict['noteval_states_shape'])))

            self.imgs = noteval_states_list
        else:
            all_states_list = np.memmap(
                self.data_dir/'sample_all.npy',
                dtype='uint8', mode='r',
                shape=(
                    tuple(dataset_dict['all_states_shape'])))
            self.imgs = all_states_list[
                :int(len(all_states_list) * train_set_ratio)
                ] if split == "train" else all_states_list[
                int(len(all_states_list) * train_set_ratio):]

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img = np.array(self.imgs[idx], dtype=np.uint8)
        if self.transforms is not None:
            img = self.transforms(img)

        return img, 0.0  # dummy data_y to prevent breaking


class MinigridGenericDatasetEval(MinigridGenericDataset):
    def __init__(
            self, data_path, split, image_size_h, image_size_w,
            train_set_ratio,
            dataset_folder_name, use_cache=False, **kwargs):
        super(MinigridGenericDatasetEval, self).__init__(
            data_path, "eval", image_size_h, image_size_w, train_set_ratio,
            dataset_folder_name, use_cache=False, **kwargs)


class MinigridGenericDatasetNoteval(
        MinigridGenericDataset):
    def __init__(
            self, data_path, split, image_size_h, image_size_w,
            train_set_ratio,
            dataset_folder_name, use_cache=False, **kwargs):
        super(MinigridGenericDatasetNoteval, self).__init__(
            data_path, "noteval", image_size_h, image_size_w, train_set_ratio,
            dataset_folder_name, use_cache=False, **kwargs)
