"""
Environment have adverarial property
This environment have 1 opponent agent only
"""
import gym
from . import gym_gomoku


class AdversarialEnv:
    def __init__(self, environment_id, opponent_policy):
        self.env = gym.make(environment_id)
        self.__opponent_policy = opponent_policy

        self.reset()

    def set_opponent_policy(self, opponent_policy):
        self.__opponent_policy = opponent_policy

    def step(self, action):
        return self.env.step(action)

    def reset(self):
        # A trick
        # Reset env and attach defined opponent policy to env
        return self.env.step(self.__opponent_policy)

    def render(self):
        return self.env.render()

    @property
    def action_space(self):
        return self.env.action_space

    @property
    def observation_space(self):
        return self.env.observation_space
