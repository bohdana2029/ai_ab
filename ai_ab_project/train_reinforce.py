


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import agent 
import environment
from agent import Agent
from environment import Environment
from agent import Agent

path = "/Users/bohdanahunka/ai_ab_project/models/agent.xml"
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

# REINFORCE algorithm implementation
class REINFORCE:
    def __init__(self, agent, env):
        super().__init__()
        self.env = env
        self.agent = agent
        self.gamma = 0.99
        #self.optimizer = optim.Adam(self.agent.parameters(), lr=0.001)     
        self.episode_rewards = []
        self.episode_log_probs = []

    
    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(device)
        with torch.no_grad():
            action = self.agent.forward(state)
        self.episode_log_probs.append(torch.log(action + 1e-10)) 
        #evaluating the probabilitions of the action 
        
        #print (self.episode_log_probs)
         # Add small value to prevent log(0)
        return action.cpu().numpy()
     

    # we need to sample an an action

'''
    def update_policy(self):
        R = 0
        policy_loss = []
        returns = []
        for r in self.episode_rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns).to(torch.device)
        for log_prob, R in zip(self.episode_log_probs, returns):
            policy_loss.append(-log_prob * R)
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        self.episode_rewards = []
        self.episode_log_probs = [] 
'''
'''
    def train(self, num_episodes):
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            done = False
            while not done:
                action = self.select_action(state)
                new_state, reward, done, _ = self.env.step(action)
                self.episode_rewards.append(reward)
                state = new_state
                episode_reward += reward
            self.update_policy()
            print(f'Episode {episode+1}/{num_episodes}, Total Reward: {episode_reward}')
    def save_model(self, path):
        torch.save(self.agent.select_action.state_dict(), path)
    def load_model(self, path):
        self.agent.select_action.load_state_dict(torch.load(path))'''  