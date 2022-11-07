import os
from itertools import permutations

import numpy as np
from minigrid.envs import DynamicObstaclesEnv
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
from PIL import Image
from tqdm.auto import tqdm
from visual_pomdp_smm.dataset.generate_uniform_dynamicobs_dataset import (
    agent_view_size, generate_all_possible_states, tile_size)

seq_len = 3
random_sample_number = 1


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

    states, states_eval, states_noteval = generate_all_possible_states()
    generated_permutations = list(permutations(states, seq_len))
    # print(generated_permutations, len(generated_permutations))

    # eval_permutations = []
    # noteval_permutations = []
    eval_label = []
    for perm in generated_permutations:
        if any(x in perm for x in states_eval):
            # eval_permutations.append(perm)
            eval_label.append(True)
        else:
            # noteval_permutations.append(perm)
            eval_label.append(False)

    env = DynamicObstaclesEnv(
        size=tile_size, n_obstacles=tile_size*2,
        agent_view_size=agent_view_size)
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)
    _, _ = env.reset()

    for i, perm in enumerate(tqdm(generated_permutations)):
        for j in range(random_sample_number):

            image_label_string = "sample_eval" if eval_label[i]\
                else "sample_noteval"
            observations = []
            for state in perm:
                observations.append(
                    generate_obs(env, state))

            concat_observations = np.hstack(observations)
            im = Image.fromarray(concat_observations)
            im.save(
                "data/SequenceDynamicObs/" + image_label_string +
                str(i*random_sample_number + j)+".png")


if __name__ == "__main__":
    main()
