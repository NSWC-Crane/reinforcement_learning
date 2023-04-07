#!/usr/bin/env python3
"""
Source code:
    https://www.hackdeploy.com/multi-armed-bandit-python-example-using-ucb/

Goal:
    promoting a new product
    text target customers a message (action)
    choose one from four alternatives (kinda like 4 slot machines)
    need to find one that gets most reward

Strategy:
    Upper-Confidence-Bound Action Selection

UCB formula components:
    t = the time (or round) we are currently at
    a = action selected (in our case the message chosen)
    Nt(a) = number of times action a was selected prior to the time t
    Qt(a) = average reward of action a prior to the time t
    c = a number greater than 0 that controls the degree of exploration
    ln t = natural logarithm of t

Decision:
    The value with the maximum UCB gets chosen at each round

Misc reference:
    https://www.kaggle.com/code/sangwookchn/reinforcement-learning-using-scikit-learn/notebook

"""

# import required libraries
import math
import random
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# plt.style.use('seaborn-whitegrid')


# generate dataset consisting of 5 columns
user = list(range(0,10000))
m1 = np.random.normal(100,10,10000)
m2 = np.random.normal(105,5,10000)
m3 = np.random.normal(95,10,10000)
m4 = np.random.normal(100,5,10000)
df = pd.DataFrame({"user":user, "m1":m1,"m2":m2,"m3":m3,"m4":m4})
df.head()

# ===========================================================================
# visualize rewards for the four messages, respecctively
# ===========================================================================
sns.boxplot(x="variable", y="value", data=pd.melt(df[['m1','m2','m3','m4']]))
plt.title("Distribution of Rewards by Message")

# Expect to pick m2 if had known. But we don't know. That's why use MAB 
# using the Upper-Confidence Bound selection method
# (1) iterate through each round
# (2) take an action (select and send a message)
# (3) see its returns and pick again
# (4) will select the best message eventually


# ===========================================================================
# Multi-Armed Bandit – UCB Method
# ===========================================================================
#Initialize Variables
N = len(df.index)       # the time (or round) 
d = 4                   # number of possible messages/slot machines/jamming techniques
Qt_a = 0
Nt_a = np.zeros(d)      #number of times action a has been selected prior to T
                        #If Nt(a) = 0, then a is considered to be a maximizing action.
c = 1                   #a number greater than 0 that controls the degree of exploration

sum_rewards = np.zeros(d) #cumulative sum of reward for a particular message

#helper variables to perform analysis
hist_t = [] #holds the natural log of each round
hist_ucb_rewards = [] #holds the history of the UCB CHOSEN cumulative rewards
hist_best_possible_rewards = [] #holds the history of OPTIMAL cumulative rewards
hist_random_choice_rewards = [] #holds the history of RANDONMLY selected actions rewards
###


# loop through n rounds represented by the variable t.
# At each round, the UCB_Values array holds the UCB values, then get reset to 0
# Then loop through each possible action.
# Choose actions that have never been selected 
# Calculate the UCB value for each action and store it in the UCB_Values array.

# At the end of each round, the action (message/slot machine/jamming technique)
# containing the maximum UCB Value gets # selected. 
# The numpy argmax function returns the index of the maximum value 
# in the array. This is stored in the action_selected variable.

#loop through no of rounds #t = time
for t in range(0,N):
    UCB_Values = np.zeros(d) #array holding the ucb values. we pick the max  
    action_selected = 0
    for a in range(0, d):
        if (Nt_a[a] > 0):
            ln_t = math.log(t) #natural log of t
            hist_t.append(ln_t) #to plot natural log of t

            #calculate the UCB
            Qt_a = sum_rewards[a]/Nt_a[a]
            ucb_value = Qt_a + c*(ln_t/Nt_a[a]) 
            UCB_Values[a] = ucb_value

        #if this equals zero, choose as the maximum. Cant divide by negative     
        elif (Nt_a[a] == 0):
            UCB_Values[a] = 1e500 #make large value
        
    #select the max UCB value
    action_selected = np.argmax(UCB_Values)

    #update Values as of round t
    Nt_a[action_selected] += 1
    reward = df.values[t, action_selected+1]
    sum_rewards[action_selected] += reward
    
# ===========================================================================    
# additional analysis - choosing a message at random in each round    
# ===========================================================================
# Comparing random algorithm to UCB algorithmm
r_ = df.values[t,[1,2,3,4]]     #get all rewards for time t to a vector
r_best = r_[np.argmax(r_)]      #select the best action

pick_random = random.randrange(d) #choose an action randomly
r_random = r_[pick_random] #np.random.choice(r_) #select reward for random action
if len(hist_ucb_rewards)>0:
    hist_ucb_rewards.append(hist_ucb_rewards[-1]+reward)
    hist_best_possible_rewards.append(hist_best_possible_rewards[-1]+r_best)
    hist_random_choice_rewards.append(hist_random_choice_rewards[-1]+r_random)
else:
    hist_ucb_rewards.append(reward)
    hist_best_possible_rewards.append(r_best)
    hist_random_choice_rewards.append(r_random)    

# ===========================================================================    
# Comparison analysis - random v. UCB   
# ===========================================================================
print("Reward if we choose randonmly {0}".format(hist_random_choice_rewards[-1]))
print("Reward of our UCB method {0}".format(hist_ucb_rewards[-1]))


# ===========================================================================    
# Which jamming technique/slot machine to apply? 
# ===========================================================================    
# Nt_a, holds the number of times each action was selected
# Expect M2 as shown before

plt.bar(['m1','m2','m3','m4'],Nt_a)
plt.title("Number of times each Message was Selected")

















