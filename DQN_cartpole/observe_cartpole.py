# packages
import time
import math
import gym
import torch
import random
import matplotlib.pyplot as plt

import numpy as np
import cv2

# birch clustering
from sklearn.cluster import Birch

from DQN_double_pytorch import DQN_double

from theta_omega import theta_omega_policy

# Demonstration
env = gym.envs.make("CartPole-v1")

# get the basic size fo the action/state space
state_size = env.observation_space.shape[0]
action_size = env.action_space.n

# Number of hidden nodes in the DQN
n_hidden = 100
# Learning rate
lr = 0.0003

# load in the model
# load_path = "trained_DQN.pt"
# model = DQN_double(state_size, action_size, n_hidden)
# model.load_state_dict(torch.load(load_path))
# model.eval()

state = env.reset()
done = False

# testing variables
num_episodes = int(math.floor(state_size*200))
max_steps = 500

q_obs = np.zeros((num_episodes*max_steps, state_size+1))
q_est = np.zeros((action_size, state_size))

index = 0

action_idx = np.ones((action_size))

# watch trained agent
for idx in range(num_episodes):

    state = env.reset()
    done = False
    total_rewards = 0

    while not done:
        action = theta_omega_policy(state)
        q_obs[index] = np.append(state, action)

        # rolling average to find the centers of the cluster
        q_est[action, :] += (state - q_est[action, :])/(action_idx[action])

        action_idx[action] += 1


        # Take action and add reward to total
        state, reward, done, _ = env.step(action)
        total_rewards += reward

        index += 1

        if done == True:
            print("Episode {}: - Score = {}".format(idx, total_rewards))
            break

bp = 1

np.save("../data/cartpole_phy_q_est_{}.npy".format(num_episodes), q_est, allow_pickle=False, fix_imports=True)
np.save("../data/cartpole_phy_q_obs_{}.npy".format(num_episodes), q_obs, allow_pickle=False, fix_imports=True)

np.savetxt("../data/cartpole_phy_q_est_{}.csv".format(num_episodes), q_est, delimiter=",")
np.savetxt("../data/cartpole_phy_q_obs_{}.csv".format(num_episodes), q_obs, delimiter=",")


# define the model
# model = Birch(threshold=0.01, n_clusters=2)
#
# q_tmp = q_est[:, 0:4]
#
# # fit the model
# model.fit(q_tmp)
#
# # assign a cluster to each example
# yhat = model.predict(q_tmp)
#
# # retrieve unique clusters
# clusters = np.unique(yhat)
#
# # create scatter plot for samples from each cluster
# for cluster in clusters:
#     # get row indexes for samples with this cluster
#     row_ix = np.where(yhat == cluster)
#     # create scatter of these samples
#     pyplot.scatter(q_tmp[row_ix, 2], q_tmp[row_ix, 3])
#
# # show the plot
# pyplot.show()

env.close()


bp = 2