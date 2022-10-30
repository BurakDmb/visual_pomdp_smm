import os

# import gymnasium as gym
from minigrid.envs import MemoryEnv
from minigrid.utils.window import Window
from minigrid.wrappers import ImgObsWrapper, RGBImgPartialObsWrapper
from PIL import Image
# import numpy as np


def redraw(obs):
    if PARTIALOBS:
        observation_ae_img = Image.fromarray(obs)
        window.show_img(observation_ae_img)
    else:
        img = env.render('rgb_array', tile_size=tile_size)
        window.show_img(img)


def key_handler(event):
    print('pressed', event.key)

    if event.key == 'escape':
        window.close()

    if event.key == 'left':
        action = env.actions.left
    elif event.key == 'right':
        action = env.actions.right
    elif event.key == 'up':
        action = env.actions.forward
    elif event.key == ' ':
        action = env.actions.toggle
    elif event.key == 'pageup':
        action = env.actions.pickup
    elif event.key == 'pagedown':
        action = env.actions.drop
    elif event.key == 'enter':
        action = env.actions.done
    else:
        return
    step(action)
    return


def step(action):
    obs, reward, terminated, truncated, info = env.step(action)
    print(env.env.env.agent_pos, env.env.env.agent_dir)
    print('step=%s, reward=%.2f' % (env.step_count, reward))

    if terminated or truncated:
        print('done!')
        obs, info = env.reset()
    redraw(obs)
    if SAVEFIG:
        window.fig.savefig(
            "manual_images/Minigrid_"+str(env.count)+".png",
            bbox_inches='tight', pad_inches=0)
    env.count += 1


tile_size = 48
window = Window("gym_minigrid - MiniGrid-Empty-Random-5x5-v0")
window.reg_key_handler(key_handler)

# env = gym.make('MiniGrid-MemoryS13-v0')
grid_size = 21
env = MemoryEnv(size=grid_size, agent_view_size=5)

env = RGBImgPartialObsWrapper(env)
env = ImgObsWrapper(env)
PARTIALOBS = True
SAVEFIG = False

obs, info = env.reset()
env.env.env.agent_pos = (1, grid_size//2)
env.env.env.agent_dir = 0
obs = env.observation(env.env.observation(env.env.env.gen_obs()))
redraw(obs)
print(env.agent_pos)
env.count = 0
if not os.path.isdir("manual_images"):
    os.makedirs("manual_images/")


def main():
    window.show(block=True)


if __name__ == "__main__":
    main()
