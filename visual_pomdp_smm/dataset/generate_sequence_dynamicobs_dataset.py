import json
import os
from itertools import permutations

import numpy as np
import ray
from minigrid.envs import DynamicObstaclesEnv
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
from tqdm.auto import tqdm

from visual_pomdp_smm.dataset.generate_uniform_dynamicobs_dataset import (
    agent_view_size, generate_all_possible_states, tile_size)

seq_len = 3
ATLEAST_N = 2
random_sample_number = 1
# Do not change the random_sample_number, it might break the code since
# the memmap does not consider random sample number.


def generate_obs(env, state):

    env.env.env.agent_start_pos = (state[0], state[1])
    env.env.env.agent_start_dir = state[2]
    obs, info = env.reset()
    return obs


def main():
    if not os.path.isdir("data/"):
        os.makedirs("data/")
    if not os.path.isdir("data/SequenceDynamicObs/"):
        os.makedirs("data/SequenceDynamicObs/")
    dataset_save_dir = "data/SequenceDynamicObs/parquet_dataset"

    assert (seq_len > 1)
    print("Started generating all possible states")

    states, states_eval, states_noteval = generate_all_possible_states()
    generated_permutations = list(permutations(states, seq_len))
    # print(generated_permutations, len(generated_permutations))

    # eval_permutations = []
    # noteval_permutations = []
    eval_label = []
    eval_counter = 0
    for perm in generated_permutations:
        if sum(x in perm for x in states_eval) >= ATLEAST_N:
            # eval_permutations.append(perm)
            eval_label.append(True)
            eval_counter += 1
        else:
            # noteval_permutations.append(perm)
            eval_label.append(False)

    env = DynamicObstaclesEnv(
        size=tile_size, n_obstacles=tile_size*2,
        agent_view_size=agent_view_size)
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)
    _, _ = env.reset()

    dataset_dict = {}
    len_total_states = len(generated_permutations)
    len_states_eval = eval_counter
    len_states_noteval = len(generated_permutations) - eval_counter

    concat_obs_shape = env.env.observation_space.spaces["image"].shape
    concat_obs_shape = (
        concat_obs_shape[0],
        concat_obs_shape[1]*seq_len,
        concat_obs_shape[2])

    all_shape = (
        len_total_states,
        *concat_obs_shape)
    eval_shape = (
        len_states_eval,
        *concat_obs_shape)
    noteval_shape = (
        len_states_noteval,
        *concat_obs_shape)

    # all_states_list = np.memmap(
    #     'data/SequenceDynamicObs/sample_all.npy',
    #     dtype='uint8', mode='write',
    #     shape=(
    #         len_total_states,
    #         *concat_obs_shape))
    dataset_dict['all_states_shape'] = all_shape

    # eval_states_list = np.memmap(
    #     'data/SequenceDynamicObs/sample_eval.npy',
    #     dtype='uint8', mode='write',
    #     shape=(
    #         len_states_eval,
    #         *concat_obs_shape))
    dataset_dict['eval_states_shape'] = eval_shape

    # noteval_states_list = np.memmap(
    #     'data/SequenceDynamicObs/sample_noteval.npy',
    #     dtype='uint8', mode='write',
    #     shape=(
    #         len_states_noteval,
    #         *concat_obs_shape))
    dataset_dict['noteval_states_shape'] = noteval_shape

    dataset_sizemb = np.prod(all_shape)/(1024*1024)
    print(
        "Uncompressed Dataset will take total ammount of: ",
        dataset_sizemb, " MB in memory.")
    mem_total_bytes = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    # Calculating the total memory, also reserving 6GB for system operations.
    mem_total_mib = mem_total_bytes/(1024.**2) - (1024*6)
    mem_total_mib = 1024*16
    if mem_total_mib > dataset_sizemb:
        dataset_partition_count = 1
        partition_file_size = dataset_sizemb
    else:
        partition_file_size = np.power(2, np.floor(np.log2(mem_total_mib)))
        dataset_partition_count = int(
            np.ceil(dataset_sizemb / partition_file_size))

    print(
        "The dataset will consist of ", dataset_partition_count,
        " partition(s), each occupying a maximum of ",
        partition_file_size, " MB.")

    prev_dataset_index = -1
    all_states_python_list = []
    eval_states_python_list = []
    noteval_states_python_list = []

    partition_arr = np.array_split(
        generated_permutations, dataset_partition_count)
    with tqdm(total=len(generated_permutations)) as pbar:
        for dataset_index, epi_array in enumerate(partition_arr):
            if prev_dataset_index != dataset_index:
                if prev_dataset_index != -1:

                    ray.init()
                    ds_eval = ray.data.from_items(
                        np.array(eval_states_python_list))
                    ds_eval = ds_eval.add_column(
                        "label", lambda df: "eval")
                    ds_noteval = ray.data.from_items(
                        np.array(noteval_states_python_list))
                    ds_noteval = ds_noteval.add_column(
                        "label", lambda df: "noteval")

                    ds_all = ds_eval.union(ds_noteval)
                    ds_all.write_parquet(dataset_save_dir)
                    del ds_eval
                    del ds_noteval
                    del ds_all
                    ray.shutdown()

                prev_dataset_index = dataset_index
                all_states_python_list = []
                eval_states_python_list = []
                noteval_states_python_list = []

            eval_i = 0
            noteval_i = 0
            for i, perm in enumerate(epi_array):
                for j in range(random_sample_number):
                    if eval_label[i]:
                        observations = []
                        for state in perm:
                            observations.append(
                                generate_obs(env, state))
                        concat_observations = np.hstack(observations)

                        all_states_python_list.append(concat_observations)
                        eval_states_python_list.append(concat_observations)
                        # all_states_list[i] = concat_observations
                        # eval_states_list[eval_i] = concat_observations
                        eval_i += 1
                    else:
                        observations = []
                        for state in perm:
                            observations.append(
                                generate_obs(env, state))
                        concat_observations = np.hstack(observations)

                        all_states_python_list.append(concat_observations)
                        noteval_states_python_list.append(concat_observations)
                        # all_states_list[i] = concat_observations
                        # noteval_states_list[noteval_i] = concat_observations
                        noteval_i += 1

                pbar.update(1)

    ray.init()
    ds_eval = ray.data.from_items(
        np.array(eval_states_python_list))
    ds_eval = ds_eval.add_column(
        "label", lambda df: "eval")
    ds_noteval = ray.data.from_items(
        np.array(noteval_states_python_list))
    ds_noteval = ds_noteval.add_column(
        "label", lambda df: "noteval")

    ds_all = ds_eval.union(ds_noteval)
    ds_all.write_parquet(dataset_save_dir)
    del ds_eval
    del ds_noteval
    del ds_all
    ray.shutdown()

    # all_states_list.flush()
    # eval_states_list.flush()
    # noteval_states_list.flush()
    json.dump(dataset_dict, open(
        "data/SequenceDynamicObs/dataset_dict.json", 'w'))

    # ds_eval = ray.data.from_items(eval_states_list)
    # ds_eval = ds_eval.add_column(
    #     "label", lambda df: "eval")
    # ds_noteval = ray.data.from_items(noteval_states_list)
    # ds_noteval = ds_noteval.add_column(
    #     "label", lambda df: "noteval")

    # ds_all = ds_eval.union(ds_noteval)
    # ds_eval.write_parquet("data/SequenceDynamicObs/parquet_eval_dataset")
    # ds_noteval.write_parquet("data/SequenceDynamicObs/parquet_noteval_dataset")
    # read_ds = ray.data.read_parquet(
    #   "data/SequenceDynamicObs/parquet_dataset")


if __name__ == "__main__":
    main()
