import json
import os
from itertools import permutations

import numpy as np
from minigrid.envs import DynamicObstaclesEnv
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
from tqdm.auto import tqdm

from visual_pomdp_smm.dataset.generate_uniform_dynamicobs_dataset import (
    agent_view_size, generate_all_possible_states, tile_size)

seq_len = 3
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
        if any(x in perm for x in states_eval):
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

    all_states_list = np.memmap(
        'data/SequenceDynamicObs/sample_all.npy',
        dtype='uint8', mode='write',
        shape=(
            len_total_states,
            *concat_obs_shape))
    dataset_dict['all_states_shape'] = all_states_list.shape

    eval_states_list = np.memmap(
        'data/SequenceDynamicObs/sample_eval.npy',
        dtype='uint8', mode='write',
        shape=(
            len_states_eval,
            *concat_obs_shape))
    dataset_dict['eval_states_shape'] = eval_states_list.shape

    noteval_states_list = np.memmap(
        'data/SequenceDynamicObs/sample_noteval.npy',
        dtype='uint8', mode='write',
        shape=(
            len_states_noteval,
            *concat_obs_shape))
    dataset_dict['noteval_states_shape'] = noteval_states_list.shape

    print(
        "Dataset will take of ammount: ",
        all_states_list.size*all_states_list.itemsize/(1024*1024), " MB")

    eval_i = 0
    noteval_i = 0
    for i, perm in enumerate(tqdm(generated_permutations)):
        for j in range(random_sample_number):
            if eval_label[i]:
                observations = []
                for state in perm:
                    observations.append(
                        generate_obs(env, state))
                concat_observations = np.hstack(observations)

                all_states_list[i] = concat_observations
                eval_states_list[eval_i] = concat_observations
                eval_i += 1
            else:
                observations = []
                for state in perm:
                    observations.append(
                        generate_obs(env, state))
                concat_observations = np.hstack(observations)
                all_states_list[i] = concat_observations
                noteval_states_list[noteval_i] = concat_observations
                noteval_i += 1

    all_states_list.flush()
    eval_states_list.flush()
    noteval_states_list.flush()
    json.dump(dataset_dict, open(
        "data/SequenceDynamicObs/dataset_dict.json", 'w'))


if __name__ == "__main__":
    main()
