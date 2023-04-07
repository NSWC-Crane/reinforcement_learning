import datetime
from pathlib import Path

from pyglet import clock
import time

import gym
from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation
from wrappers import ResizeObservation, SkipFrame

import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY, SIMPLE_MOVEMENT, COMPLEX_MOVEMENT

from nes_py.wrappers import JoypadSpace
from nes_py._image_viewer import ImageViewer

# from metrics import MetricLogger
from observation_logger import observation_logger

# the sentinel value for "No Operation"
_NOP = 0

env = gym_super_mario_bros.make('SuperMarioBros-1-1-v0')
# env = JoypadSpace(env, COMPLEX_MOVEMENT)

# env = SkipFrame(env, skip=4)
env = GrayScaleObservation(env, keep_dim=False)
env = ResizeObservation(env, shape=84)
env = TransformObservation(env, f=lambda x: x / 255.)
# env = FrameStack(env, num_stack=4)
env.reset()


def play_human(env, viewer, callback=None):
    """
    Play the environment using keyboard as a human.
    Args:
        env: the initialized gym environment to play
        callback: a callback to receive output from the environment
    Returns:
        None
    """
    # ensure the observation space is a box of pixels
    assert isinstance(env.observation_space, gym.spaces.box.Box)

    # ensure the observation space is either B&W pixels or RGB Pixels
    obs_s = env.observation_space
    is_bw = len(obs_s.shape) == 2
    is_rgb = len(obs_s.shape) == 3 and obs_s.shape[2] in [1, 3]
    assert is_bw or is_rgb

    # create a done flag for the environment
    done = True

    # prepare frame rate limiting
    target_frame_duration = 1 / env.metadata['video.frames_per_second']
    last_frame_time = 0
    # start the main game loop
    try:
        state = env.reset()

        while True:
            current_frame_time = time.time()

            # limit frame rate
            if last_frame_time + target_frame_duration > current_frame_time:
                continue

            # save frame beginning time for next refresh
            last_frame_time = current_frame_time

            # clock tick
            clock.tick()

            # reset if the environment is done
            if done:
                viewer.show(env.unwrapped.screen)
                break
                # done = False
                # state = env.reset()

            # unwrap the action based on pressed relevant keys
            action = keys_to_action.get(viewer.pressed_keys, _NOP)
            next_state, reward, done, _ = env.step(action)

            viewer.show(env.unwrapped.screen)

            # pass the observation data through the callback
            if callback is not None:
                callback.record(action, state)

            state = next_state

            # shutdown if the escape key is pressed
            if viewer.is_escape_pressed:
                break

    except KeyboardInterrupt:
        pass

    # viewer.close()


# get the mapping of keyboard keys to actions in the environment
if hasattr(env, 'get_keys_to_action'):
    keys_to_action = env.get_keys_to_action()
elif hasattr(env.unwrapped, 'get_keys_to_action'):
    keys_to_action = env.unwrapped.get_keys_to_action()
else:
    raise ValueError('env has no get_keys_to_action method')

# create the image viewer
viewer = ImageViewer(
    env.spec.id if env.spec is not None else env.__class__.__name__,
    # env.observation_space.shape[0], # height
    # env.observation_space.shape[1], # width
    500, 500,
    monitor_keyboard=True,
    relevant_keys=set(sum(map(list, keys_to_action.keys()), []))
    )


#------------------- START --------------------

save_dir = Path('checkpoints') / datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
obs_logger = observation_logger(save_dir)

obs_logger.init_episode(0)

play_human(env, viewer, callback=obs_logger)

episodes = 100

# for e in range(episodes):

    # state = env.reset()

viewer.close()

bp = 1
