# Run `pip install "gymnasium[classic-control]"` for this example.
import gymnasium as gym
from gymnasium import spaces    
import numpy as np

import mujoco
import cv2
import glfw
from agent import Agent



class Environment(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 30}

# action space is the thurst of the 4 propellers,
#  which can be between 0 and 180 degrees
  

##################### 
    def __init__(self, render_mode = None):
        super(Environment, self).__init__()
        self.model = mujoco.MjModel.from_xml_path('/Users/bohdanahunka/ai_ab_project/models/agent.xml')
        self.data = mujoco.MjData(self.model)
        self.reward_count = 0
        self.action_space = spaces.Box(low=0, high=200, shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32) 
 ##############  
    def reset(self, seed = None, options = None):
        self.data.qpos[:] = [0.0, 0.0, 0.3, 1.0, 0.0, 0.0, 0.0]
        self.data.qvel[:] = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        self.data.ctrl[:] = [0.0, 0.0, 0.0, 0.0]
        self.reward_count = 0
        mujoco.mj_forward(self.model, self.data)
        observation = self.get_state()
        obs = np.array(observation)
        info = {}
        return obs, info
    
################# current state of the agent in the environemnt
    def get_state(self):
        pos = self.data.qpos[:7]      
        vel = self.data.qvel[:6]    
        state = np.concatenate([pos, vel])
        state = np.array(state)
        return state

###################      
    def step(self, action):
        currect_state = self.get_state()
        self.data.ctrl[:] = action
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)
        new_state = self.get_state()
        reward = self.reward_function( currect_state, new_state)
        terminated = False
        truncated = False
        info = {}
        new_state = np.array(new_state)
        reward = np.array(reward)
        
        # print("New state after action:", new_state)
        return new_state, reward, terminated, truncated, info
       

    def targer_location(self):
        self.target_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, 'target')
        target_position = self.data.geom_xpos[self.target_id]
        return target_position

    def reward_function(self, state, new_state):
        reward = 0

        desired_height = 4
         
        old_z_position = state[2]
        new_z_position = new_state[2]
        old_x_position = state[0]
        new_x_position = new_state[0]
        old_y_position = state[1]
        new_y_position = new_state[1]

        old_distance = abs(desired_height - old_z_position)
        new_distance = abs(desired_height - new_z_position)

        if new_distance < old_distance:
            reward += 1
        elif (new_distance > old_distance):
            reward -= 1


        reward += new_state[10] 


        if abs(new_state[8]) < 0.5 or abs(new_state[9]) < 0.5:
            reward += 1
        else:
            reward -= 1
       

        if abs(new_state[10]) > 3:
            reward += 10
        else:
            reward -= 1
    # Base reward: closer to target = better
        

    # Extra reward if moving toward target
       
        if new_x_position - old_x_position > 0.3:
            reward -= 1
        if new_y_position - old_y_position > 0.3:
            reward -= 1 
        
        if new_z_position >= 4:
            reward -= 10

        return reward
    
    def render(self):
        return 0        

    def close(self):
        pass




        #des
        # 4. Avoid spinning too much
        #
        ##
        #
        # Staying for stay near desired height


        # reward for being stable in the air
        #
        #
        #
        # reward for being able to fly to the target in a certain amount of time

        #
        #
        # reward for varying the thurst of the propellers in a certain way to be able to fly to the target
        #
        #
        #

        # 6. Use smooth motor/thrust changes
        ##
        #
        #crash penalty

        #
        #
        #
        #
        #

     
        
        # i want reward to be able to fly to a certain height in 3d stace
        # i want the reawrd to be able to fly to a certain target in the 3d space
        # based on the reward will be updateed 
     
    # i need to somehow basically say that the the drone should be able to fly to a certain height and stay there, 
    # and also fly to a certain target in the 3d space, and based on the reward will be updateed
 

  
    