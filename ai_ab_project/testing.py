
import environment
from environment import Environment


env = environment.Environment()
print("Sample action:", env.action_space.sample())
print("Sample observation:", env.observation_space.sample())