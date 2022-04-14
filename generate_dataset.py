from gym_minigrid.envs.empty import EmptyEnv
from gym_minigrid.wrappers import RGBImgObsWrapper
import os
from PIL import Image


class EmptyRandomEnv16x16(EmptyEnv):
    def __init__(self):
        super().__init__(size=16, agent_start_pos=None)


tile_size = 8

env = EmptyRandomEnv16x16()
env = RGBImgObsWrapper(env)

if not os.path.isdir("dataset_images/"):
    os.makedirs("dataset_images/")


def main():
    for i in range(1024*100):
        _ = env.reset()
        img = env.render('rgb_array', tile_size=tile_size)
        im = Image.fromarray(img)
        im.save("dataset_images/minigrid_"+str(i)+".png")


if __name__ == "__main__":
    main()
