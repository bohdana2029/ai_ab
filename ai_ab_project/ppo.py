import gymnasium as gym 

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from environment import Environment as env
import register_env


import mujoco
import mujoco.viewer


env = gym.make('Drone')

model = PPO.load("ppo_drone", env= env)




# need to convert my environment into gymnasium

# remove to demonstrate saving and loading

model.learn(total_timesteps=10000)
model.save("ppo_drone")



with mujoco.viewer.launch_passive(env.unwrapped.model, env.unwrapped.data) as viewer:
    
    viewer.cam.distance = 20.0
    viewer.cam.elevation = -60.0
    
    while viewer.is_running():
              
         
            action, _ = model.predict(obs)
            obs, rewards, terminared, trucated, info = env.step(action)
           
            viewer.sync()




