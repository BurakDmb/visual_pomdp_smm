import json
import os
from itertools import permutations

import numpy as np
import ray
from minigrid.core.world_object import Ball, Key
from minigrid.envs import MemoryEnv
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
from tqdm.auto import tqdm

from visual_pomdp_smm.dataset.generate_uniform_memory_dataset import (
    generate_all_possible_states, tile_size)

seq_len = 3
ATLEAST_N = 2


def generate_obs(env, state, start_room_obj, hallway_end, other_objs):

    env.env.env.agent_pos = (state[0], state[1])
    env.env.env.agent_dir = state[2]

    env.env.env.grid.set(
        1, tile_size // 2 - 1, start_room_obj("green"))
    pos0 = (hallway_end + 1, tile_size // 2 - 2)
    pos1 = (hallway_end + 1, tile_size // 2 + 2)
    env.env.env.grid.set(*pos0, other_objs[0]("green"))
    env.env.env.grid.set(*pos1, other_objs[1]("green"))
    obs = env.observation(env.env.observation(
                    env.env.env.gen_obs()))
    return obs


def main():
    if not os.path.isdir("data/"):
        os.makedirs("data/")
    if not os.path.isdir("data/SequenceMemory/"):
        os.makedirs("data/SequenceMemory/")
    dataset_save_dir = "data/SequenceMemory/parquet_dataset"

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

    env = MemoryEnv(size=tile_size, agent_view_size=5)
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
        4*len_total_states,
        *concat_obs_shape)
    eval_shape = (
        4*len_states_eval,
        *concat_obs_shape)
    noteval_shape = (
        4*len_states_noteval,
        *concat_obs_shape)

    # The number 4 comes from the 4 combination of key-door.
    # all_states_list = np.memmap(
    #     'data/SequenceMemory/sample_all.npy',
    #     dtype='uint8', mode='write',
    #     shape=(
    #         4*len_total_states,
    #         *concat_obs_shape))
    dataset_dict['all_states_shape'] = all_shape

    # eval_states_list = np.memmap(
    #     'data/SequenceMemory/sample_eval.npy',
    #     dtype='uint8', mode='write',
    #     shape=(
    #         4*len_states_eval,
    #         *concat_obs_shape))
    dataset_dict['eval_states_shape'] = eval_shape

    # noteval_states_list = np.memmap(
    #     'data/SequenceMemory/sample_noteval.npy',
    #     dtype='uint8', mode='write',
    #     shape=(
    #         4*len_states_noteval,
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

                # For each key and door combination (total of 4)
                # 0: Start:Key, Doors: Ball-Key
                observations = []
                start_room_obj = Key
                other_objs = [Ball, Key]
                hallway_end = tile_size - 3
                for state in perm:
                    observations.append(
                        generate_obs(
                            env, state,
                            start_room_obj,
                            hallway_end, other_objs))

                concat_observations0 = np.hstack(observations)

                # 1: Start:Key, Doors: Key-Ball
                observations = []
                start_room_obj = Key
                other_objs = [Key, Ball]
                hallway_end = tile_size - 3
                for state in perm:
                    observations.append(
                        generate_obs(
                            env, state,
                            start_room_obj,
                            hallway_end, other_objs))

                concat_observations1 = np.hstack(observations)

                # 2: Start:Ball, Doors: Ball-Key
                observations = []
                start_room_obj = Ball
                other_objs = [Ball, Key]
                hallway_end = tile_size - 3
                for state in perm:
                    observations.append(
                        generate_obs(
                            env, state,
                            start_room_obj,
                            hallway_end, other_objs))

                concat_observations2 = np.hstack(observations)

                # 3: Start:Ball, Doors: Key-Ball
                observations = []
                start_room_obj = Ball
                other_objs = [Key, Ball]
                hallway_end = tile_size - 3
                for state in perm:
                    observations.append(
                        generate_obs(
                            env, state,
                            start_room_obj,
                            hallway_end, other_objs))

                concat_observations3 = np.hstack(observations)

                if eval_label[i]:
                    all_states_python_list.append(concat_observations0)
                    all_states_python_list.append(concat_observations1)
                    all_states_python_list.append(concat_observations2)
                    all_states_python_list.append(concat_observations3)

                    eval_states_python_list.append(concat_observations0)
                    eval_states_python_list.append(concat_observations1)
                    eval_states_python_list.append(concat_observations2)
                    eval_states_python_list.append(concat_observations3)

                    # all_states_list[4*i+0] = concat_observations0
                    # all_states_list[4*i+1] = concat_observations1
                    # all_states_list[4*i+2] = concat_observations2
                    # all_states_list[4*i+3] = concat_observations3
                    # eval_states_list[4*eval_i+0] = concat_observations0
                    # eval_states_list[4*eval_i+1] = concat_observations1
                    # eval_states_list[4*eval_i+2] = concat_observations2
                    # eval_states_list[4*eval_i+3] = concat_observations3
                    eval_i += 1
                else:
                    all_states_python_list.append(concat_observations0)
                    noteval_states_python_list.append(concat_observations0)

                    all_states_python_list.append(concat_observations1)
                    noteval_states_python_list.append(concat_observations1)

                    all_states_python_list.append(concat_observations2)
                    noteval_states_python_list.append(concat_observations2)

                    all_states_python_list.append(concat_observations3)
                    noteval_states_python_list.append(concat_observations3)
                    # all_states_list[4*i+0] = concat_observations0
                    # all_states_list[4*i+1] = concat_observations1
                    # all_states_list[4*i+2] = concat_observations2
                    # all_states_list[4*i+3] = concat_observations3
                    # noteval_states_list[4*noteval_i+0] = concat_observations0
                    # noteval_states_list[4*noteval_i+1] = concat_observations1
                    # noteval_states_list[4*noteval_i+2] = concat_observations2
                    # noteval_states_list[4*noteval_i+3] = concat_observations3
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
        "data/SequenceMemory/dataset_dict.json", 'w'))

    # ds_eval = ray.data.from_items(eval_states_list)
    # ds_eval = ds_eval.add_column(
    #     "label", lambda df: "eval")
    # ds_noteval = ray.data.from_items(noteval_states_list)
    # ds_noteval = ds_noteval.add_column(
    #     "label", lambda df: "noteval")

    # ds_all = ds_eval.union(ds_noteval)
    # ds_eval.write_parquet("data/SequenceMemory/parquet_eval_dataset")
    # ds_noteval.write_parquet("data/SequenceMemory/parquet_noteval_dataset")
    # read_ds = ray.data.read_parquet("data/SequenceMemory/parquet_dataset")


if __name__ == "__main__":
    main()
