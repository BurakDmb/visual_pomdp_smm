# from gym_minigrid.envs.empty import EmptyEnv
from gym_minigrid.wrappers import RGBImgObsWrapper
import gym
import os
from PIL import Image


# class EmptyRandomEnv(EmptyEnv):
#     def __init__(self):
#         super().__init__(size=8, agent_start_pos=None)


def main():
    tile_size = 8

    # env = EmptyRandomEnv()

    env = gym.make('MiniGrid-Empty-Random-6x6-v0')
    env = RGBImgObsWrapper(env)

    if not os.path.isdir("dataset_images/"):
        os.makedirs("dataset_images/")

    for i in range(1024*100):
        _ = env.reset()
        img = env.render('rgb_array', tile_size=tile_size, highlight=False)
        im = Image.fromarray(img)
        im.save("dataset_images/minigrid_"+str(i)+".png")


if __name__ == "__main__":
    main()
