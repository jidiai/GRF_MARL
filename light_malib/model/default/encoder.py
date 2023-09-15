from gym.spaces import Box, Discrete
import numpy as np
import gym

class Encoder:
    def __init__(self, action_spaces, observation_spaces, state_space):

        self._action_space = action_spaces
        self._observation_space = observation_spaces
        self._state_space = state_space

    def encode(self, state):
        # obs=np.array([self._policy.state_index(state)],dtype=int)
        # print(self._policy.state_index(state))
        obs = state
        action_mask = np.ones(self._action_space.n, dtype=np.float32)
        return obs, action_mask

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def state_space(self):
        return self._state_space