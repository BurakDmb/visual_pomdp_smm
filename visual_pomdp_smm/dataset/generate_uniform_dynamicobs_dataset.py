import os

from minigrid.envs import DynamicObstaclesEnv
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
from PIL import Image
from tqdm.auto import tqdm

# Total sample number is 230400
tile_size = 21
sample_per_episode = (tile_size-2)*(tile_size-2)*4 - 4
epi_number = int(230400/sample_per_episode)
k = tile_size//2
agent_view_size = 5


def main():

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

    for epi in tqdm(range(epi_number)):
        i = 0
        for state in states_eval:
            env.env.env.agent_start_pos = (state[0], state[1])
            env.env.env.agent_start_dir = state[2]
            obs, info = env.reset()
            # obs = env.observation(env.env.observation(env.env.env.gen_obs()))
            im = Image.fromarray(obs)
            im.save(
                "data/UniformDynamicObs/sample_eval" +
                str(epi*len(states)+i)+".png")
            i += 1
        for state in states_noteval:
            env.env.env.agent_start_pos = (state[0], state[1])
            env.env.env.agent_start_dir = state[2]
            obs, info = env.reset()
            # obs = env.observation(env.env.observation(env.env.env.gen_obs()))
            im = Image.fromarray(obs)
            im.save(
                "data/UniformDynamicObs/sample_noteval" +
                str(epi*len(states)+i)+".png")
            i += 1


if __name__ == "__main__":
    main()
