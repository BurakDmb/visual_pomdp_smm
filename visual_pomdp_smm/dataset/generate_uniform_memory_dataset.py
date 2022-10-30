import os

from minigrid.envs import MemoryEnv
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
from PIL import Image
from tqdm.auto import tqdm

# Total sample number is 230400
tile_size = 21
sample_per_episode = ((tile_size-2) + 5)*4
epi_number = int(230400/sample_per_episode)
k = tile_size//2


def main():

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

    env = MemoryEnv(size=tile_size, agent_view_size=5)

    # env = RGBImgObsWrapper(env)
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)

    if not os.path.isdir("data/"):
        os.makedirs("data/")
    if not os.path.isdir("data/UniformMemory/"):
        os.makedirs("data/UniformMemory/")

    # Count the random object prob. Commented out, only for debug purposes.
    # counter = 0
    for epi in tqdm(range(epi_number)):
        obs, info = env.reset()
        i = 0

        # Count the random object prob. Commented out, only for debug purposes.
        # if env.success_pos == (tile_size-2, k+1):
        #     counter += 1

        for state in states_eval:
            env.env.env.agent_pos = (state[0], state[1])
            env.env.env.agent_dir = state[2]
            obs = env.observation(env.env.observation(env.env.env.gen_obs()))
            im = Image.fromarray(obs)
            im.save(
                "data/UniformMemory/sample_eval" +
                str(epi*len(states)+i)+".png")
            i += 1
        for state in states_noteval:
            env.env.env.agent_pos = (state[0], state[1])
            env.env.env.agent_dir = state[2]
            obs = env.observation(env.env.observation(env.env.env.gen_obs()))
            im = Image.fromarray(obs)
            im.save(
                "data/UniformMemory/sample_noteval" +
                str(epi*len(states)+i)+".png")
            i += 1
    # Count the random object prob. Commented out, only for debug purposes.
    # print("Random object prob." + str(counter/epi_number))


if __name__ == "__main__":
    main()
