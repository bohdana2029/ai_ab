# Investigating the Performance of Policy-Gradient Reinforcement Learning Algorithms for Quadrotor Control Task in MuJoCo Simulation

This project investigates and compares two Reinforcement Learning algorithms for drone control in a simulated environment: optimised REINFORCE and Proximal Policy Optimization (PPO).

**Link to robot sim working (trained by PPO):** https://www.youtube.com/shorts/3ev-ldHn4Ds

**Agent and Environment**

The simulation is implemented in MuJoCo using an XML file. The agent is represented as a simplified drone model consisting of a central body, four propeller arms, and four motor actuators.

The 13-dimensional state representation obtained
from the environment is used as the input to this neural network. The architecture consists of an 13-value input layer, three hidden layers with ReLU activation with 64 neurons each, and an output layer with a sigmoid activation function. The hidden layers allow the network to learn non-linear relationships between the state variables and motor actions.

The sigmoid activation constrains the output values to the range [0,1]. They are scaled by a factor of 175, which was selected as a tuning hyperparameter to produce suitable thrust values for the drone simulation. 

The environment is designed to be compatible with Gymnasium-based algorithms by following the required environment structure. This allows the implementation of PPO from the Stable-Baselines3 library.

