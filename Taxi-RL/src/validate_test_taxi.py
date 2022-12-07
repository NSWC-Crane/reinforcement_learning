import numpy as np
import gym
# import pygame
import random

# create Taxi environment
env = gym.make('Taxi-v3')
env.reset()
env.step(1)
env.render(mode='human')

# get the basic size fo the action/state space
state_size = env.observation_space.n
action_size = env.action_space.n

# testing variables
num_episodes = state_size*20
max_steps = 200  # per episode

# load in teh saved qtable from the training
qtable = np.load("../data/taxi_qtable.npy")

q_est = np.load("../data/taxi_q_est_1.npy")

# Challenge:
#  1. we don't know the state numbering
#  2. we do know the actions, but not the correct order
#

count = 0
acc = 0

# watch trained agent
for idx in range(num_episodes):

    state = env.reset()
    done = False
    rewards = 0

    print("Episode {}".format(idx))

    for s in range(max_steps):

        action = np.argmax(qtable[state, :])
        est_action = np.argmax(q_est[state, :])

        if est_action == action:
            acc += 1

        new_state, reward, done, info = env.step(action)

        rewards += reward
        # env.render(mode='human')
        state = new_state

        count += 1

        if done == True:
            print("  Step {}".format(s+1))
            print("  Score: {}".format(rewards))
            break

bp = 1

print("Accuracy: {}".format(acc/count))

env.close()

