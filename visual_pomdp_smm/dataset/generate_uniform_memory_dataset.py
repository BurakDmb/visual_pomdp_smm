import json
import os

import numpy as np
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

    dataset_dict = {}
    len_total_states = len(states)
    len_states_eval = len(states_eval)
    len_states_noteval = len(states_noteval)

    all_states_list = np.memmap(
        'data/UniformMemory/sample_all.npy',
        dtype='uint8', mode='write',
        shape=(
            len_total_states * epi_number,
            *env.env.observation_space.spaces["image"].shape))
    dataset_dict['all_states_shape'] = all_states_list.shape

    eval_states_list = np.memmap(
        'data/UniformMemory/sample_eval.npy',
        dtype='uint8', mode='write',
        shape=(
            len_states_eval * epi_number,
            *env.env.observation_space.spaces["image"].shape))
    dataset_dict['eval_states_shape'] = eval_states_list.shape

    noteval_states_list = np.memmap(
        'data/UniformMemory/sample_noteval.npy',
        dtype='uint8', mode='write',
        shape=(
            len_states_noteval * epi_number,
            *env.env.observation_space.spaces["image"].shape))
    dataset_dict['noteval_states_shape'] = noteval_states_list.shape

    for epi in tqdm(range(epi_number)):
        obs, info = env.reset()
        i = 0

        for j, state in enumerate(states_eval):
            env.env.env.agent_pos = (state[0], state[1])
            env.env.env.agent_dir = state[2]
            obs = env.observation(env.env.observation(env.env.env.gen_obs()))
            all_states_list[len_total_states * epi + i] = obs
            eval_states_list[len_states_eval * epi + j] = obs
            i += 1

        for j, state in enumerate(states_noteval):
            env.env.env.agent_pos = (state[0], state[1])
            env.env.env.agent_dir = state[2]
            obs = env.observation(env.env.observation(env.env.env.gen_obs()))
            all_states_list[len_total_states * epi + i] = obs
            noteval_states_list[len_states_noteval * epi + j] = obs
            i += 1

    all_states_list.flush()
    eval_states_list.flush()
    noteval_states_list.flush()
    json.dump(dataset_dict, open(
        "data/UniformMemory/dataset_dict.json", 'w'))


if __name__ == "__main__":
    main()
