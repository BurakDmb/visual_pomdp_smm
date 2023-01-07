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
import ray
import pandas as pd
# import matplotlib.pyplot as plt


# plt.rcParams['figure.dpi'] = 200
# torch.manual_seed(0)
# device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_data(use_ray, data_dir, split, dataset_dict):

    if use_ray:
        def filter_eval(batch: pd.DataFrame) -> pd.DataFrame:
            filter_rows = batch[batch['label'] != "eval"].index
            batch = batch.drop(filter_rows, axis=0)
            return batch

        def filter_noteval(batch: pd.DataFrame) -> pd.DataFrame:
            filter_rows = batch[batch['label'] != "noteval"].index
            batch = batch.drop(filter_rows, axis=0)
            return batch

        data = None
        read_ds = ray.data.read_parquet(data_dir+"/parquet_dataset")
        if split == 'eval':
            data = read_ds.map_batches(filter_eval)
        elif split == "noteval":
            data = read_ds.map_batches(filter_noteval)
        else:
            data = read_ds
        data = data.to_torch()

    # This else block kept used for backwards compatibility.
    else:
        if split == 'eval':
            data = np.memmap(
                data_dir/'sample_eval.npy',
                dtype='uint8', mode='r',
                shape=(
                    tuple(dataset_dict['eval_states_shape'])))
        elif split == "noteval":
            data = np.memmap(
                data_dir/'sample_noteval.npy',
                dtype='uint8', mode='r',
                shape=(
                    tuple(dataset_dict['noteval_states_shape'])))
        else:
            data = np.memmap(
                data_dir/'sample_all.npy',
                dtype='uint8', mode='r',
                shape=(
                    tuple(dataset_dict['all_states_shape'])))
    return data


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
            eval_states_list = load_data(
                use_ray=False, data_dir=self.data_dir,
                split=split, dataset_dict=dataset_dict)

            if use_cache:
                self.imgs = np.array(eval_states_list)
            else:
                self.imgs = eval_states_list[:]

        elif split == "noteval":
            noteval_states_list = load_data(
                use_ray=False, data_dir=self.data_dir,
                split=split, dataset_dict=dataset_dict)

            if use_cache:
                self.imgs = np.array(noteval_states_list)
            else:
                self.imgs = noteval_states_list[:]

        elif split == "all":
            all_states_list = load_data(
                use_ray=False, data_dir=self.data_dir,
                split=split, dataset_dict=dataset_dict)

            if use_cache:
                self.imgs = np.array(all_states_list)
            else:
                self.imgs = all_states_list[:]
        else:
            all_states_list = load_data(
                use_ray=False, data_dir=self.data_dir,
                split=split, dataset_dict=dataset_dict)

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
            dataset_folder_name, use_cache=use_cache, **kwargs)


class MinigridGenericDatasetNoteval(
        MinigridGenericDataset):
    def __init__(
            self, data_path, split, image_size_h, image_size_w,
            train_set_ratio,
            dataset_folder_name, use_cache=False, **kwargs):
        super(MinigridGenericDatasetNoteval, self).__init__(
            data_path, "noteval", image_size_h, image_size_w, train_set_ratio,
            dataset_folder_name, use_cache=use_cache, **kwargs)
