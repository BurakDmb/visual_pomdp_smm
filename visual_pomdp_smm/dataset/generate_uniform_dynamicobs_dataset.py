import json
import os

import numpy as np
from minigrid.envs import DynamicObstaclesEnv
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
from tqdm.auto import tqdm

# Total sample number is 230400
tile_size = 9
sample_per_episode = (tile_size-2)*(tile_size-2)*4 - 4
epi_number = int(230400/sample_per_episode)
k = tile_size//2

# Fixed, Could not be changed
agent_view_size = 5


def generate_all_possible_states():
    assert agent_view_size == 5
    # Defining all traversable states for memory env.
    # Total of: (tile_size-2)*(tile_size-2)*4 - 4
    states = []

    for direction in range(4):
        for i in range(tile_size-2):
            for j in range(tile_size-2):
                if not (i+1 == tile_size-2 and j+1 == tile_size-2):
                    states.append((i+1, j+1, direction))

    states_eval = []

    states_eval.append((tile_size-2, tile_size-6, 1))
    states_eval.append((tile_size-2, tile_size-5, 1))
    states_eval.append((tile_size-2, tile_size-4, 1))
    states_eval.append((tile_size-2, tile_size-3, 1))

    states_eval.append((tile_size-3, tile_size-6, 1))
    states_eval.append((tile_size-3, tile_size-5, 1))
    states_eval.append((tile_size-3, tile_size-4, 1))
    states_eval.append((tile_size-3, tile_size-3, 1))
    states_eval.append((tile_size-3, tile_size-2, 1))

    states_eval.append((tile_size-4, tile_size-6, 1))
    states_eval.append((tile_size-4, tile_size-5, 1))
    states_eval.append((tile_size-4, tile_size-4, 1))
    states_eval.append((tile_size-4, tile_size-3, 1))
    states_eval.append((tile_size-4, tile_size-2, 1))

    states_eval.append((tile_size-6, tile_size-2, 0))
    states_eval.append((tile_size-5, tile_size-2, 0))
    states_eval.append((tile_size-4, tile_size-2, 0))
    states_eval.append((tile_size-3, tile_size-2, 0))

    states_eval.append((tile_size-6, tile_size-3, 0))
    states_eval.append((tile_size-5, tile_size-3, 0))
    states_eval.append((tile_size-4, tile_size-3, 0))
    states_eval.append((tile_size-3, tile_size-3, 0))
    states_eval.append((tile_size-2, tile_size-3, 0))

    states_eval.append((tile_size-6, tile_size-4, 0))
    states_eval.append((tile_size-5, tile_size-4, 0))
    states_eval.append((tile_size-4, tile_size-4, 0))
    states_eval.append((tile_size-3, tile_size-4, 0))
    states_eval.append((tile_size-2, tile_size-4, 0))

    states_eval.append((tile_size-4, tile_size-2, 3))
    states_eval.append((tile_size-3, tile_size-2, 3))

    states_eval.append((tile_size-2, tile_size-4, 2))
    states_eval.append((tile_size-2, tile_size-3, 2))

    states_noteval = [x for x in states if x not in states_eval]

    return states, states_eval, states_noteval


def main():

    states, states_eval, states_noteval = generate_all_possible_states()

    env = DynamicObstaclesEnv(
        size=tile_size, n_obstacles=tile_size*2,
        agent_view_size=agent_view_size)

    # env = RGBImgObsWrapper(env)
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)

    if not os.path.isdir("data/"):
        os.makedirs("data/")
    if not os.path.isdir("data/UniformDynamicObs/"):
        os.makedirs("data/UniformDynamicObs/")

    dataset_dict = {}
    len_total_states = len(states)
    len_states_eval = len(states_eval)
    len_states_noteval = len(states_noteval)

    all_states_list = np.memmap(
        'data/UniformDynamicObs/sample_all.npy',
        dtype='uint8', mode='write',
        shape=(
            len_total_states * epi_number,
            *env.env.observation_space.spaces["image"].shape))
    dataset_dict['all_states_shape'] = all_states_list.shape

    eval_states_list = np.memmap(
        'data/UniformDynamicObs/sample_eval.npy',
        dtype='uint8', mode='write',
        shape=(
            len_states_eval * epi_number,
            *env.env.observation_space.spaces["image"].shape))
    dataset_dict['eval_states_shape'] = eval_states_list.shape

    noteval_states_list = np.memmap(
        'data/UniformDynamicObs/sample_noteval.npy',
        dtype='uint8', mode='write',
        shape=(
            len_states_noteval * epi_number,
            *env.env.observation_space.spaces["image"].shape))
    dataset_dict['noteval_states_shape'] = noteval_states_list.shape

    for epi in tqdm(range(epi_number)):
        i = 0

        for j, state in enumerate(states_eval):
            env.env.env.agent_start_pos = (state[0], state[1])
            env.env.env.agent_start_dir = state[2]
            obs, info = env.reset()
            # obs = env.observation(env.env.observation(env.env.env.gen_obs()))
            all_states_list[len_total_states * epi + i] = obs
            eval_states_list[len_states_eval * epi + j] = obs
            i += 1

        for j, state in enumerate(states_noteval):
            env.env.env.agent_start_pos = (state[0], state[1])
            env.env.env.agent_start_dir = state[2]
            obs, info = env.reset()
            # obs = env.observation(env.env.observation(env.env.env.gen_obs()))
            all_states_list[len_total_states * epi + i] = obs
            noteval_states_list[len_states_noteval * epi + j] = obs
            i += 1

    all_states_list.flush()
    eval_states_list.flush()
    noteval_states_list.flush()
    json.dump(dataset_dict, open(
        "data/UniformDynamicObs/dataset_dict.json", 'w'))


if __name__ == "__main__":
    main()
