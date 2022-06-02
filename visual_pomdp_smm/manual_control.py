from gym_minigrid.window import Window
from gym_minigrid.wrappers import RGBImgObsWrapper
import os
import gym


def redraw(obs):
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
    obs, reward, done, info = env.step(action)
    print('step=%s, reward=%.2f' % (env.step_count, reward))

    if done:
        print('done!')
        obs = env.reset()
    redraw(obs)
    window.fig.savefig(
        "manual_images/Minigrid_"+str(env.count)+".png",
        bbox_inches='tight', pad_inches=0)
    env.count += 1


tile_size = 32
window = Window("gym_minigrid - MiniGrid-Empty-Random-5x5-v0")
window.reg_key_handler(key_handler)

env = gym.make('MiniGrid-Empty-Random-6x6-v0')
env = RGBImgObsWrapper(env)
obs = env.reset()
redraw(obs)
env.count = 0
if not os.path.isdir("manual_images"):
    os.makedirs("manual_images/")


def main():
    window.show(block=True)


if __name__ == "__main__":
    main()
