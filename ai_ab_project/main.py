

import matplotlib
from matplotlib import path
import mujoco
import mujoco.viewer
import time
import math
import os 
from environment import Environment
from agent import Agent
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim 
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import reinforce
from reinforce import REINFORCE
matplotlib.use('Agg')

from visualisation_tools import plot_rewards




path = "/Users/bohdanahunka/ai_ab_project/models/agent.xml"
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

environment = Environment()
agent = Agent(input_size=13, hidden_size=64, output_size=4).to(device)
reinforce = REINFORCE(agent, environment)

# for collection trajectories 

state = environment.get_state() 
rewards_on_episodes = []
rewards_on_steps = []   
num_episodes = 50000
num_steps = 3000

terminate = False
# type: ignore


with mujoco.viewer.launch_passive(environment.model, environment.data) as viewer:
    viewer.cam.distance = 20.0
    viewer.cam.elevation = -60.0


    while viewer.is_running() and not terminate:
    
        for episode in range(num_episodes):
            print("Episode number",episode+1)
            state= environment.reset()

    
            for step in range(num_steps):

                #generation a trajectory for an episode
                target = environment.targer_location()
                action_agent = agent.select_action(state)      
                action_sample= reinforce.sample_action(action_agent)
                new_state = environment.step( action_sample.cpu().numpy())
                reward = environment.reward_function(state, new_state)
                state = new_state
                ###################################################################
                #summing rewards for an episode
                environment.reward_count += reward
                rewards_on_steps.append(reward)
            
                # reinforce loop ###########
                log_probs = reinforce.evaluate_action(action_sample)
                a,b,c,d = reinforce.collect_trajectory(state, action_sample, reward, log_probs)
              
                #update = reinforce.update_policy()
            
                # can change target location to be random in the environment, 
                # but for now its fixed
                viewer.sync()

           # print("states",len(a))
           # print("the length od actions",len(b))
           # print("the length od rewards",len(c))
           # print(" the length of the logs", len(d))
            G = reinforce.discounted_returns(c)
            policy_loss = reinforce.collect_loss(d,G)
            reinforce.update_policy(policy_loss)
            rewards_on_episodes.append(environment.reward_count)
            



            reinforce.reset_reinforce_data()
            terminate = True

            #print( reinforce.episode_states)
            #print("Discounted Returns: {}".format(reinforce.discounted_returns(rewards_on_steps)))
            #reinforce.save_model('models/agent_model.pth')
            #print(" Eposide: {} score: {}".format(episode, environment.reward_count))
    rewards_on_episodes = np.array(rewards_on_episodes)
    #print(rewards_on_episodes)
    num = []
    for a in range( num_episodes):
        num.append(a)
    num = np.array(num)
    #print(num)

    plot_rewards(rewards_on_episodes, num)
    
        
   

            


#print("Discounted Returns: {}".format(reinforce.discounted_returns(rewards_on_steps  )  ))     
    



    
    

    