# from gym_minigrid.wrappers import RGBImgObsWrapper
from gym_minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
import gym
import os
from PIL import Image
from tqdm.auto import tqdm


def main():
    tile_size = 13
    class_key_to_all_ratio = 0.1
    total_samples = 1024*100

    # env = gym.make('MiniGrid-DoorKey-8x8-v0')
    # env = gym.make('MiniGrid-MemoryS17Random-v0')
    env = gym.make('MiniGrid-MemoryS13-v0')

    # env = RGBImgObsWrapper(env)
    env = RGBImgPartialObsWrapper(env)
    env = ImgObsWrapper(env)

    if not os.path.isdir("data/"):
        os.makedirs("data/")
    if not os.path.isdir("data/MinigridKey/"):
        os.makedirs("data/MinigridKey/")

    for i in tqdm(range(total_samples)):
        obs = env.reset()
        # Full environment
        img_orig = env.render(
            'rgb_array', tile_size=tile_size, highlight=False)
        im_orig = Image.fromarray(img_orig)

        # Class 2- No starting object
        im_other = Image.fromarray(obs)

        env.step(0)
        obs, _, _, _ = env.step(0)
        # Class 1 - Starting object
        im_key = Image.fromarray(obs)

        if i < int(total_samples*class_key_to_all_ratio):
            im_key.save("data/MinigridKey/minigridkey_key"+str(i)+".png")
        else:
            im_other.save("data/MinigridKey/minigridkey_other"+str(i)+".png")
        im_orig.save("data/MinigridKey/_minigridkey_full_env_"+str(i)+".png")


if __name__ == "__main__":
    main()
