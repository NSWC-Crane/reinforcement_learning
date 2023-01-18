#!/usr/bin/env python3
"""
Created on Mon Aug 30 13:23:45 2021

Source code:
    https://www.tensorflow.org/agents/tutorials/1_dqn_tutorial
    
    
Goal:
    Train a DQN (Deep Q Networks) agent on the Cartpole environment using the TF-Agents library.

Walk through:
    Reinforcement Learning (RL) pipeline for training, evaluation and data collection.
    
To use the imageio ffmpeg plugin you need to 'pip install imageio-ffmpeg'    
Ref.: https://imageio.readthedocs.io/en/stable/


installed the following dependencies via Anaconda Prompt:
pip install 'imageio==2.4.0'
pip install pyvirtualdisplay
pip install tf-agents
pip install pyglet


Main website: https://imageio.github.io
Imageio is a Python library that provides an easy interface to read and write 
a wide range of image data, including animated images, volumetric data, and 
scientific formats. It is cross-platform, runs on Python 3.5+, and is easy to install.


Ref.: https://github.com/openai/gym/wiki/CartPole-v0
    Episode Termination -
    Pole Angle is more than ±12°
    Cart Position is more than ±2.4 (center of the cart reaches the edge of the display)
    Episode length is greater than 200 (500 for v1).
    That is why the step-size is 200
    
"""

from __future__ import absolute_import, division, print_function

import base64
import imageio
import IPython
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import pyvirtualdisplay

import tensorflow as tf

from tf_agents.agents.dqn import dqn_agent
from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import sequential
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.specs import tensor_spec
from tf_agents.utils import common

import time
import os
from datetime import date

today = date.today()

# Check if TensorFlow is using GPU acceleration 
len(tf.config.list_physical_devices('GPU'))


# Set up a virtual display for rendering OpenAI gym environments.
# display = pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()

tf.version.VERSION

# cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
# Hyperparameters
# cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

num_iterations = 20000 # @param {type:"integer"}

initial_collect_steps = 100  # @param {type:"integer"}
collect_steps_per_iteration =   1# @param {type:"integer"}
replay_buffer_max_length = 100000  # @param {type:"integer"}

batch_size = 64  # @param {type:"integer"}
learning_rate = 1e-3  # @param {type:"number"}
log_interval = 200  # @param {type:"integer"}; logs loss

num_eval_episodes = 10  # @param {type:"integer"}
eval_interval = 1000  # @param {type:"integer"}; reports Average Return 


# ########################################################################################
# Environment
# ########################################################################################
'''
In Reinforcement Learning (RL), an environment represents the task or problem to be solved. 
Standard environments can be created in TF-Agents using tf_agents.environments suites. 
TF-Agents has suites for loading environments from sources such as the OpenAI Gym, Atari, and DM Control.
Load the CartPole environment from the OpenAI Gym suite.
'''
env_name = 'CartPole-v0'
env = suite_gym.load(env_name)

'''
A free-swinging pole is attached to a cart. The goal is to move the cart right or left 
in order to keep the pole pointing up.

Install pyglet directly via 'pip install pyglet' using Anaconda Prompt
'''
env.reset()
# PIL.Image.fromarray(env.render())


'''
The environment.step method takes an action in the environment and returns a TimeStep tuple 
containing the next observation of the environment and the reward for the action.

The time_step_spec() method returns the specification for the TimeStep tuple. 
Its observation attribute shows the shape of observations, the data types, and the ranges 
of allowed values. The reward attribute shows the same details for the reward.
'''
print('\nObservation Spec:')
print(env.time_step_spec().observation)

print('\nReward Spec:')
print(env.time_step_spec().reward)

'''
The action_spec() method returns the shape, data types, and allowed values of valid actions.
'''
print('\nAction Spec:')
print(env.action_spec())


# cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
# The Cartpole environment:
# cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

'''
observation is an array of 4 floats:
    the position and velocity of the cart
    the angular position and velocity of the pole
reward is a scalar float value
action is a scalar integer with only two possible values:
    0 — "move left"
    1 — "move right"
'''
time_step = env.reset()
print('\nTime step:')
print(time_step)

action = np.array(1, dtype=np.int32)

next_time_step = env.step(action)
print('\nNext time step:')
print(next_time_step)

'''
Usually two environments are instantiated: one for training and one for evaluation.
'''
train_py_env = suite_gym.load(env_name)
eval_py_env = suite_gym.load(env_name)

# cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc
# Data type conversion
# cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc

'''
The Cartpole environment is converted from Python to TensorFlow using the 
TFPyEnvironment wrapper.

The TFPyEnvironment converts Numpy arrays to Tensors to make it compatible 
with Tensorflow agents and policies.
'''
train_env = tf_py_environment.TFPyEnvironment(train_py_env)
eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)


# ########################################################################################
# Agent
# ########################################################################################

'''
The algorithm used to solve an RL problem is represented by an Agent. 
TF-Agents provides standard implementations of a variety of Agents, including:
DQN (used in this tutorial)
REINFORCE
DDPG
TD3
PPO
SAC.

The DQN agent can be used in any environment which has a discrete action space.

At the heart of a DQN Agent is a QNetwork, a neural network model that can learn to predict 
QValues (expected returns) for all actions, given an observation from the environment.

tf_agents.networks are used to create a QNetwork. The network will consist of a sequence 
of tf.keras.layers.Dense layers, where the final layer will have 1 output for each possible action.
'''
fc_layer_params = (100, 50)
action_tensor_spec = tensor_spec.from_spec(env.action_spec())
num_actions = action_tensor_spec.maximum - action_tensor_spec.minimum + 1

# Define a helper function to create Dense layers configured with the right
# activation and kernel initializer.
def dense_layer(num_units):
  return tf.keras.layers.Dense(
      num_units,
      activation=tf.keras.activations.relu,
      kernel_initializer=tf.keras.initializers.VarianceScaling(
          scale=2.0, mode='fan_in', distribution='truncated_normal'))

# QNetwork consists of a sequence of Dense layers followed by a dense layer
# with `num_actions` units to generate one q_value per available action as
# it's output.
dense_layers = [dense_layer(num_units) for num_units in fc_layer_params]
q_values_layer = tf.keras.layers.Dense(
    num_actions,
    activation=None,
    kernel_initializer=tf.keras.initializers.RandomUniform(
        minval=-0.03, maxval=0.03),
    bias_initializer=tf.keras.initializers.Constant(-0.2))
q_net = sequential.Sequential(dense_layers + [q_values_layer])

'''
Use tf_agents.agents.dqn.dqn_agent to instantiate a DqnAgent. 

In addition to the time_step_spec, action_spec and the QNetwork, the agent constructor 
also requires an optimizer (in this case, AdamOptimizer), a loss function, 
and an integer step counter.
'''
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

train_step_counter = tf.Variable(0)

agent = dqn_agent.DqnAgent(
    train_env.time_step_spec(),
    train_env.action_spec(),
    q_network=q_net,
    optimizer=optimizer,
    td_errors_loss_fn=common.element_wise_squared_loss,
    train_step_counter=train_step_counter)

agent.initialize()


# ########################################################################################
# Policies
# ########################################################################################

'''
A policy defines the way an agent acts in an environment. 

The goal of reinforcement learning is to train the underlying model until the 
policy produces the desired outcome.

In this tutorial:
The desired outcome is keeping the pole balanced upright over the cart.
The policy returns an action (left or right) for each time_step observation.

Agents contain two policies:
agent.policy — The main policy that is used for evaluation and deployment.
agent.collect_policy — A second policy that is used for data collection.
'''
eval_policy = agent.policy
collect_policy = agent.collect_policy

'''
Policies can be created independently of agents. 

For example, use tf_agents.policies.random_tf_policy to create a policy which 
will randomly select an action for each time_step.
'''
random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                train_env.action_spec())

'''
Call the policy.action(time_step) method to get an action from a policy. 

The time_step contains the observation from the environment. 

This method returns a PolicyStep, which is a named tuple with three components:
action — the action to be taken (in this case, 0 or 1)
state — used for stateful (that is, RNN-based) policies
info — auxiliary data, such as log probabilities of actions
'''

example_environment = tf_py_environment.TFPyEnvironment(suite_gym.load('CartPole-v0'))

time_step = example_environment.reset()

random_policy.action(time_step)

# ########################################################################################
# Metrics and Evaluation
# ########################################################################################

'''
The most common metric used to evaluate a policy is the average return. 
The return is the sum of rewards obtained while running a policy in an 
environment for an episode. Several episodes are run, creating an average return.

The following function computes the average return of a policy, given 
the policy, environment, and a number of episodes.
'''
def compute_avg_return(environment, policy, num_episodes=10):

  total_return = 0.0
  for _ in range(num_episodes):

    time_step = environment.reset()
    episode_return = 0.0

    while not time_step.is_last():
      action_step = policy.action(time_step)
      time_step = environment.step(action_step.action)
      episode_return += time_step.reward
    total_return += episode_return

  avg_return = total_return / num_episodes
  return avg_return.numpy()[0]


# See also the metrics module for standard implementations of different metrics.
# https://github.com/tensorflow/agents/tree/master/tf_agents/metrics

'''
Running this computation on the random_policy shows a baseline performance in the environment.
'''
compute_avg_return(eval_env, random_policy, num_eval_episodes)

avg_return_random = np.round(compute_avg_return(eval_env, random_policy, num_eval_episodes),2)
print(f'\nAverage return based on random policy: {avg_return_random}')

avg_return_agent = np.round(compute_avg_return(eval_env, agent.policy, num_eval_episodes),2)
print(f'\nAverage return based on agent policy: {avg_return_agent}')



# ########################################################################################
# Replay Buffer
# ########################################################################################

'''
The replay buffer keeps track of data collected from the environment. 

tf_agents.replay_buffers.tf_uniform_replay_buffer.TFUniformReplayBuffer is the most common method.

The constructor requires the specs for the data it will be collecting, the batch size and maximum buffer length. 
The specs for the data are available from the agent using the collect_data_spec method.  
'''
replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
    data_spec=agent.collect_data_spec,
    batch_size=train_env.batch_size,
    max_length=replay_buffer_max_length)

'''
For most agents, collect_data_spec is a named tuple called Trajectory, containing 
the specs for observations, actions, rewards, and other items.
'''
print('\nagent.collect_data_spec:\n',agent.collect_data_spec)

print('\nagent.collect_data_spec._fields:\n',agent.collect_data_spec._fields)


# ########################################################################################
# Data Collection
# ########################################################################################

'''
Execute the random policy in the environment for a few steps, recording the data in the replay buffer.
'''
def collect_step(environment, policy, buffer):
  time_step = environment.current_time_step()
  action_step = policy.action(time_step)
  next_time_step = environment.step(action_step.action)
  traj = trajectory.from_transition(time_step, action_step, next_time_step)

  # Add trajectory to the replay buffer
  buffer.add_batch(traj)

def collect_data(env, policy, buffer, steps):
  for _ in range(steps):
    collect_step(env, policy, buffer)

collect_data(train_env, random_policy, replay_buffer, initial_collect_steps)

# This loop is so common in RL, that we provide standard implementations. 
# For more details see tutorial 4 or the drivers module.
# https://github.com/tensorflow/agents/blob/master/docs/tutorials/4_drivers_tutorial.ipynb 
# https://www.tensorflow.org/agents/api_docs/python/tf_agents/drivers


'''
The replay buffer is now a collection of Trajectories.
'''
# For the curious:
# Uncomment to peel one of these off and inspect it.
# iter(replay_buffer.as_dataset()).next()

'''
The agent needs access to the replay buffer. This is provided by creating an 
iterable tf.data.Dataset pipeline which will feed data to the agent.

Each row of the replay buffer only stores a single observation step. But since 
the DQN Agent needs both the current and next observation to compute the loss, 
the dataset pipeline will sample two adjacent rows for each item in the batch (num_steps=2).

This dataset is also optimized by running parallel calls and prefetching data.
'''
# Dataset generates trajectories with shape [Bx2x...]
dataset = replay_buffer.as_dataset(
    num_parallel_calls=3, 
    sample_batch_size=batch_size, 
    num_steps=2).prefetch(3)


dataset

iterator = iter(dataset)
print(iterator)

# For the curious:
# Uncomment to see what the dataset iterator is feeding to the agent.
# Compare this representation of replay data 
# to the collection of individual trajectories shown earlier.

# iterator.next()


# ########################################################################################
# Training the agent
# ########################################################################################

'''
Two things must happen during the training loop:

collect data from the environment
use that data to train the agent's neural network(s)

This example also periodicially evaluates the policy and prints the current score.

The following will take ~6 minutes to run.
'''
time_1 = time.time()

# (Optional) Optimize by wrapping some of the code in a graph using TF function.
agent.train = common.function(agent.train)

# Reset the train step
agent.train_step_counter.assign(0)

# Evaluate the agent's policy once before training.
avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
returns = [avg_return]

for _ in range(num_iterations):

  # Collect a few steps using collect_policy and save to the replay buffer.
  collect_data(train_env, agent.collect_policy, replay_buffer, collect_steps_per_iteration)

  # Sample a batch of data from the buffer and update the agent's network.
  experience, unused_info = next(iterator)
  train_loss = agent.train(experience).loss

  step = agent.train_step_counter.numpy()

  if step % log_interval == 0:
    print('step = {0}: loss = {1}'.format(step, train_loss))

  if step % eval_interval == 0:
    avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    print('step = {0}: Average Return = {1}'.format(step, avg_return))
    returns.append(avg_return)

time_train = np.round(time.time() - time_1, 2)
print(f'\ntime_train: {time_train}')  # 396.0 seconds

# ########################################################################################
# Visualization
# ########################################################################################

'''
Use matplotlib.pyplot to chart how the policy improved during training.

One iteration of Cartpole-v0 consists of 200 time steps. The environment gives 
a reward of +1 for each step the pole stays up, so the maximum return for one 
episode is 200. The charts shows the return increasing towards that maximum 
each time it is evaluated during training. (It may be a little unstable and 
not increase monotonically each time.)
'''
# Save plot files
os.chdir('C:/Users/astai/Documents/Crane2020_and_Beyond/#Research/2_Experiments/3_Plots/RL/')

plt.figure()
iterations = range(0, num_iterations + 1, eval_interval)
plt.plot(iterations, returns)
plt.ylabel('Average Return')
plt.xlabel('Iterations')
plt.ylim(top=250)
plt.savefig("".join(['rl_cartpole_', str(today), '.png']), dpi=300)
plt.show()


# ########################################################################################
# Videos
# ########################################################################################

"""

'''
Watch an agent actually performing a task in an environment.

First, create a function to embed videos in the notebook.
'''
def embed_mp4(filename):
  '''Embeds an mp4 file in the notebook.'''
  video = open(filename,'rb').read()
  b64 = base64.b64encode(video)
  tag = '''
  <video width="640" height="480" controls>
    <source src="data:video/mp4;base64,{0}" type="video/mp4">
  Your browser does not support the video tag.
  </video>'''.format(b64.decode())

  return IPython.display.HTML(tag)



'''
Now iterate through a few episodes of the Cartpole game with the agent. 
The underlying Python environment (the one "inside" the TensorFlow environme
wrapper) provides a render() method, which outputs an image of the environment 
state. These can be collected into a video.
'''
def create_policy_eval_video(policy, filename, num_episodes=5, fps=30):
  filename = filename + ".mp4"
  with imageio.get_writer(filename, fps=fps) as video:
    for _ in range(num_episodes):
      time_step = eval_env.reset()
      video.append_data(eval_py_env.render())
      while not time_step.is_last():
        action_step = policy.action(time_step)
        time_step = eval_env.step(action_step.action)
        video.append_data(eval_py_env.render())
        video.close()
        return embed_mp4(filename)

# Save image files
os.chdir('C:/Users/astai/Documents/Crane2020_and_Beyond/#Research/2_Experiments/4_images/RL/')
                     

# To use the imageio ffmpeg plugin you need to 'pip install imageio-ffmpeg'

create_policy_eval_video(agent.policy, "trained-agent")


# For fun, compare the trained agent (above) to an agent moving randomly. (It does not do as well.)
create_policy_eval_video(random_policy, "random-agent")

# ?? How to close the video windows??
"""
