
import numpy as np

# https://towardsdatascience.com/how-to-beat-the-cartpole-game-in-5-lines-5ab4e738c93f
# https://github.com/jianxu305/openai-gym-docker/blob/main/example/Solving_CartPole_in_5_Lines.ipynb
def theta_omega_policy(obs):
    theta, w = obs[2:4]
    if abs(theta) < 0.03:
        return 0 if w < 0 else 1
    else:
        return 0 if theta < 0 else 1
