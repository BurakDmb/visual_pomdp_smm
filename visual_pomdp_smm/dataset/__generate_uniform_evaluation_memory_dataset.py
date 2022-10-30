import os

from minigrid.envs import MemoryEnv
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
from PIL import Image
from tqdm.auto import tqdm

tile_size = 13
sample_per_episode = 22
epi_number = 320
k = tile_size//2


def main():

    # Defining all traversable states for memory env.
    # Total of: ((tile_size-2) + 5)*4
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

    env = MemoryEnv(size=tile_size, agent_view_size=5)

    # env = RGBImgObsWrapper(env)
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)

    if not os.path.isdir("data/"):
        os.makedirs("data/")
    if not os.path.isdir("data/UniformEvaluationMemory/"):
        os.makedirs("data/UniformEvaluationMemory/")

    for epi in tqdm(range(epi_number)):
        obs, info = env.reset()
        for i, state in enumerate(states_eval):
            env.env.env.agent_pos = (state[0], state[1])
            env.env.env.agent_dir = state[2]
            obs = env.observation(env.env.observation(env.env.env.gen_obs()))
            im = Image.fromarray(obs)
            im.save(
                "data/UniformEvaluationMemory/sample" +
                str(epi*len(states_eval)+i)+".png")


if __name__ == "__main__":
    main()
