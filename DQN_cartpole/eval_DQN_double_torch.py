# packages
import time
import math
import gym
import torch
import random
import matplotlib.pyplot as plt
from PIL import Image
from IPython.display import clear_output
import torchvision.transforms as T
import numpy as np
import cv2


from DQN_double_pytorch import DQN_double

# Demonstration
env = gym.envs.make("CartPole-v1")

# Number of states
n_state = env.observation_space.shape[0]
# Number of actions
n_action = env.action_space.n

# Number of hidden nodes in the DQN
n_hidden = 100
# Learning rate
lr = 0.0003


# load in the model
load_path = "trained_DQN.pt"
model = DQN_double(n_state, n_action, n_hidden)
model.load_state_dict(torch.load(load_path))
model.eval()


state = env.reset()
done = False

total = 0

while not done:
    q_values = model.target_predict(state)
    action = torch.argmax(q_values).item()

    # Take action and add reward to total
    next_state, reward, done, _ = env.step(action)
    total += reward

    img = env.render(mode="rgb_array")

    cv2.imshow("image", img)
    cv2.waitKey(10)

    state = next_state

print("total reward: {}".format(total))

bp = 1

