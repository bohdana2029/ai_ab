import mujoco
import time
import math
import os 
import torch
import numpy as np
import torch.nn as nn
from torch.distributions import Categorical
import torch.optim as optim 
from torch.utils.data import DataLoader
device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"


class Agent(nn.Module):
####################### neural network architecture for the agent
    def __init__(self, input_size, hidden_size, output_size):
        super(Agent, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)

###################### forward pass through the network
    def forward(self, x):
        #just outputting action between 0 and 1, and scaling it to 0 and 200, so the drone has enough
        #thust to fly in my simulation ( not a very good practive, but its working:))
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        action = torch.sigmoid(self.fc4(x)) 
        action = action * 200.0 # Scale output to [0, 180]
        return action

######################### select action based on the current state of the environment
    def select_action(self, state):
        state = torch.tensor(state, dtype=torch.float32).to(device)
        action = self.forward(state)     
        return action
  
