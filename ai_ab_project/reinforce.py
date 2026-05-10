


import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import agent 
import environment
from agent import Agent
from environment import Environment

path = "/Users/bohdanahunka/ai_ab_project/models/agent.xml"
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"

# REINFORCE algorithm implementation
class REINFORCE:

# objective to maximize the expected return by updatng the neural netwoek policy 

    def __init__(self, agent, env):
        super().__init__()
        self.env = env
        self.agent = agent
        self.gamma = 0.99
        self.optimizer = optim.Adam(self.agent.parameters(), lr=0.001) 
        # for collection trajectories    
        self.episode_rewards = []
        self.episode_action_actions = []
        self.episode_states = []
        self.episode_log_probs = []


    def reset_reinforce_data(self):
        self.episode_rewards =[]
        self.episode_action_actions = []
        self.episode_log_probs = []
        self.episode_rewards = []


    def collect_trajectory(self, state, action, reward, log_probs):
        self.episode_states.append(state)
        self.episode_action_actions.append(action)
        self.episode_rewards.append(reward)
        self.episode_log_probs.append(log_probs)
        return self.episode_states,self.episode_action_actions, self.episode_rewards, self.episode_log_probs

    
    def sample_action(self, action):
        # taking an action from the dictribution
        std = torch.ones_like(action) * 50.0 
        self.dist = torch.distributions.Normal(action, std)
        action = self.dist.sample()
        # only when sending to the envisonment
        return action
     

    def evaluate_action(self, action):
        log_probs = self.dist.log_prob(action)
        # not sure, sso confusing!
        log_probs = log_probs.sum()
        
        #print(log_probs)
        return log_probs  
    

    def discounted_returns(self, episode_rewards, gamma=0.99):
        #sum of discounted rewards,
        returns = []
        
        for t in range(len(episode_rewards)):
            discounted = 0
            for k,r in enumerate(episode_rewards[t:]):
                discounted += (gamma**k)*r
            returns.append(discounted)
  
        returns = torch.tensor(returns, dtype=torch.float32).to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9) # Normalization
        #print(returns)
        return returns
    
    def collect_loss(self, log_probs,discounts):
        policy_loss = []
        for  log_prob, discount \
            in zip(log_probs,discounts):
            loss = -log_prob*discount
            policy_loss.append(loss)

        #needs changing
        policy_loss = torch.stack(policy_loss).sum()
        return policy_loss
          

    def update_policy(self, policy_loss):
        
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
  

        
        





    
    # what action controls the thurst of the 4 propellers, which can be between 0 and 180 
        #robs = torch.nn.functional.softmax(x, dim=-1)
        
        #
        #log_prob = torch.log(torch.tensor(probs) + 1e-10)  # Add small value to prevent log(0)
        #print("log_prob:", log_prob)
        #print("Action probabilities:", probs)

        # sampling an action from the probability distribution output by the network
        #dist = torch.distributions.Normal(probs)  # Adding some exploration noise
        #action = dist.sample()
  
         # Ensure action is within valid range
    


    
     

'''
   
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