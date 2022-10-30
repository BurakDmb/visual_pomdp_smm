import os

from minigrid.envs import DynamicObstaclesEnv
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
from PIL import Image
from tqdm.auto import tqdm

tile_size = 13
sample_per_episode = 32
epi_number = 256


def main():

    # Defining all traversable states for memory env.
    # Total of: (tile_size-2)*(tile_size-2)*4 - 4
    states_eval = []

    states_eval.append((11, 7, 1))
    states_eval.append((11, 8, 1))
    states_eval.append((11, 9, 1))
    states_eval.append((11, 10, 1))

    states_eval.append((10, 7, 1))
    states_eval.append((10, 8, 1))
    states_eval.append((10, 9, 1))
    states_eval.append((10, 10, 1))
    states_eval.append((10, 11, 1))

    states_eval.append((9, 7, 1))
    states_eval.append((9, 8, 1))
    states_eval.append((9, 9, 1))
    states_eval.append((9, 10, 1))
    states_eval.append((9, 11, 1))

    states_eval.append((7, 11, 0))
    states_eval.append((8, 11, 0))
    states_eval.append((9, 11, 0))
    states_eval.append((10, 11, 0))

    states_eval.append((7, 10, 0))
    states_eval.append((8, 10, 0))
    states_eval.append((9, 10, 0))
    states_eval.append((10, 10, 0))
    states_eval.append((11, 10, 0))

    states_eval.append((7, 9, 0))
    states_eval.append((8, 9, 0))
    states_eval.append((9, 9, 0))
    states_eval.append((10, 9, 0))
    states_eval.append((11, 9, 0))

    states_eval.append((9, 11, 3))
    states_eval.append((10, 11, 3))

    states_eval.append((11, 9, 2))
    states_eval.append((11, 10, 2))

    env = DynamicObstaclesEnv(
        size=tile_size, n_obstacles=tile_size*2, agent_view_size=5)

    # env = RGBImgObsWrapper(env)
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)

    if not os.path.isdir("data/"):
        os.makedirs("data/")
    if not os.path.isdir("data/UniformEvaluationDynamicObs/"):
        os.makedirs("data/UniformEvaluationDynamicObs/")

    for epi in tqdm(range(epi_number)):
        for i, state in enumerate(states_eval):
            env.env.env.agent_start_pos = (state[0], state[1])
            env.env.env.agent_start_dir = state[2]
            obs, info = env.reset()
            # obs = env.observation(env.env.observation(env.env.env.gen_obs()))
            im = Image.fromarray(obs)
            im.save(
                "data/UniformEvaluationDynamicObs/sample" +
                str(epi*len(states_eval)+i)+".png")


if __name__ == "__main__":
    main()
