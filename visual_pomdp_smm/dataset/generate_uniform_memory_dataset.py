import json
import os

import numpy as np
import ray
from minigrid.envs import MemoryEnv
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
from tqdm.auto import tqdm

# Total sample number is 230400
tile_size = 21
sample_per_episode = ((tile_size-2) + 5)*4
epi_number = int(230400/sample_per_episode)
k = tile_size//2


def generate_all_possible_states():
    # Defining all traversable states for memory env.
    # Total of: ((tile_size-2) + 5)*4
    states = []
    for direction in range(4):
        for i in range(tile_size-2):
            states.append((i+1, k, direction))
        states.append((1, k+1, direction))
        states.append((2, k-1, direction))
        states.append((2, k+1, direction))
        states.append((3, k-1, direction))
        states.append((3, k+1, direction))

    states_eval = []
    states_eval.append((1, k, 0))
    states_eval.append((1, k, 2))
    states_eval.append((1, k, 3))

    states_eval.append((2, k, 2))
    states_eval.append((2, k, 3))

    states_eval.append((3, k, 2))
    states_eval.append((3, k, 3))

    states_eval.append((1, k+1, 0))
    states_eval.append((1, k+1, 2))
    states_eval.append((1, k+1, 3))

    states_eval.append((2, k+1, 2))
    states_eval.append((2, k+1, 3))

    states_eval.append((3, k+1, 2))
    states_eval.append((3, k+1, 3))

    states_eval.append((2, k-1, 1))
    states_eval.append((2, k-1, 2))
    states_eval.append((2, k-1, 3))

    states_eval.append((3, k-1, 1))
    states_eval.append((3, k-1, 2))
    states_eval.append((3, k-1, 3))

    states_eval.append((4, k, 2))
    states_eval.append((5, k, 2))

    states_eval.append((tile_size-2, k, 0))
    states_eval.append((tile_size-2, k, 1))
    states_eval.append((tile_size-2, k, 2))
    states_eval.append((tile_size-2, k, 3))

    states_noteval = [x for x in states if x not in states_eval]

    return states, states_eval, states_noteval


def main():

    states, states_eval, states_noteval = generate_all_possible_states()

    env = MemoryEnv(size=tile_size, agent_view_size=5)

    # env = RGBImgObsWrapper(env)
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)

    if not os.path.isdir("data/"):
        os.makedirs("data/")
    if not os.path.isdir("data/UniformMemory/"):
        os.makedirs("data/UniformMemory/")
    dataset_save_dir = "data/UniformMemory/parquet_dataset"

    dataset_dict = {}
    len_total_states = len(states)
    len_states_eval = len(states_eval)
    len_states_noteval = len(states_noteval)

    all_shape = (
        len_total_states * epi_number,
        *env.env.observation_space.spaces["image"].shape)
    eval_shape = (
        len_states_eval * epi_number,
        *env.env.observation_space.spaces["image"].shape)

    noteval_shape = (
        len_states_noteval * epi_number,
        *env.env.observation_space.spaces["image"].shape)

    # all_states_list = np.memmap(
    #     'data/UniformMemory/sample_all.npy',
    #     dtype='uint8', mode='write',
    #     shape=all_shape)
    dataset_dict['all_states_shape'] = all_shape

    # eval_states_list = np.memmap(
    #     'data/UniformMemory/sample_eval.npy',
    #     dtype='uint8', mode='write',
    #     shape=eval_shape)
    dataset_dict['eval_states_shape'] = eval_shape

    # noteval_states_list = np.memmap(
    #     'data/UniformMemory/sample_noteval.npy',
    #     dtype='uint8', mode='write',
    #     shape=noteval_shape)
    dataset_dict['noteval_states_shape'] = noteval_shape

    dataset_sizemb = np.prod(all_shape)/(1024*1024)
    print(
        "Uncompressed Dataset will take total ammount of: ",
        dataset_sizemb, " MB in memory.")
    mem_total_bytes = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    # Calculating the total memory, also reserving 6GB for system operations.
    mem_total_mib = mem_total_bytes/(1024.**2) - (1024*6)
    # mem_total_mib = 512
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
    # all_states_python_list = []
    eval_states_python_list = []
    noteval_states_python_list = []
    partition_arr = np.array_split(
        range(epi_number), dataset_partition_count)
    with tqdm(total=epi_number) as pbar:
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
                # all_states_python_list = []
                eval_states_python_list = []
                noteval_states_python_list = []

            for epi in epi_array:
                obs, info = env.reset()
                i = 0

                for j, state in enumerate(states_eval):
                    env.env.env.agent_pos = (state[0], state[1])
                    env.env.env.agent_dir = state[2]
                    obs = env.observation(
                        env.env.observation(env.env.env.gen_obs()))
                    # all_states_python_list.append(obs)
                    eval_states_python_list.append(obs)
                    # all_states_list[len_total_states * epi + i] = obs
                    # eval_states_list[len_states_eval * epi + j] = obs
                    i += 1

                for j, state in enumerate(states_noteval):
                    env.env.env.agent_pos = (state[0], state[1])
                    env.env.env.agent_dir = state[2]
                    obs = env.observation(
                        env.env.observation(env.env.env.gen_obs()))

                    # all_states_python_list.append(obs)
                    noteval_states_python_list.append(obs)
                    # all_states_list[len_total_states * epi + i] = obs
                    # noteval_states_list[len_states_noteval * epi + j] = obs
                    i += 1
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
        "data/UniformMemory/dataset_dict.json", 'w'))

    # ds_eval = ray.data.from_items(eval_states_list)
    # ds_eval = ds_eval.add_column(
    #     "label", lambda df: "eval")
    # ds_noteval = ray.data.from_items(noteval_states_list)
    # ds_noteval = ds_noteval.add_column(
    #     "label", lambda df: "noteval")

    # ds_all = ds_eval.union(ds_noteval)
    # ds_eval.write_parquet("data/UniformMemory/parquet_eval_dataset")
    # ds_noteval.write_parquet("data/UniformMemory/parquet_noteval_dataset")
    # read_ds = ray.data.read_parquet("data/UniformMemory/parquet_dataset")


if __name__ == "__main__":
    main()
