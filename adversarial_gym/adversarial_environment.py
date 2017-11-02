"""
Environment have adverarial property
This environment have 1 opponent agent only
"""
import gym
from . import gym_gomoku


class AdversarialEnv:
    def __init__(self, environment_id, opponent_policy=None):
        self.__env = gym.make(environment_id)
        self.__opponent_policy = opponent_policy

        self.reset()

    @property
    def opponent_policy(self):
        return self.__opponent_policy

    @opponent_policy.setter
    def opponent_policy(self, opponent_policy):
        self.__opponent_policy = opponent_policy

    def swap_role(self):
        if (self.__env.player_color == 'black'):
            self.__env.player_color = 'white'
        else:
            self.__env.player_color = 'black'

    def step(self, action):
        return self.__env.step(action)

    def reset(self):
        # A trick
        # Reset env and attach defined opponent policy to env
        return self.__env.step(self.__opponent_policy)

    def render(self):
        return self.__env.render()

    @property
    def action_space(self):
        return self.__env.action_space

    @property
    def observation_space(self):
        return self.__env.observation_space
