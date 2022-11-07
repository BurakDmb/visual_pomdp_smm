import os
from itertools import permutations

import numpy as np
from minigrid.core.world_object import Ball, Key
from minigrid.envs import MemoryEnv
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
from PIL import Image
from tqdm.auto import tqdm
from visual_pomdp_smm.dataset.generate_uniform_memory_dataset import (
    generate_all_possible_states, tile_size)

seq_len = 3


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

    env = MemoryEnv(size=tile_size, agent_view_size=5)
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)
    _, _ = env.reset()

    for i, perm in enumerate(tqdm(generated_permutations)):
        image_label_string = "sample_eval" if eval_label[i]\
            else "sample_noteval"

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

        concat_observations = np.hstack(observations)
        im = Image.fromarray(concat_observations)
        im.save(
            "data/SequenceMemory/" + image_label_string +
            str(4*i + 0)+".png")

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

        concat_observations = np.hstack(observations)
        im = Image.fromarray(concat_observations)
        im.save(
            "data/SequenceMemory/" + image_label_string +
            str(4*i + 1)+".png")

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

        concat_observations = np.hstack(observations)
        im = Image.fromarray(concat_observations)
        im.save(
            "data/SequenceMemory/" + image_label_string +
            str(4*i + 2)+".png")

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

        concat_observations = np.hstack(observations)
        im = Image.fromarray(concat_observations)
        im.save(
            "data/SequenceMemory/" + image_label_string +
            str(4*i + 3)+".png")


if __name__ == "__main__":
    main()
