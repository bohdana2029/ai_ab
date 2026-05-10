from gymnasium.envs.registration import register
from environment import Environment

register(
    id="Drone_1",
    entry_point="environment:Environment",
    max_episode_steps = 20000,
)
register(
    id="Drone",
    entry_point="environment:Environment",
    max_episode_steps = 2000,
)

