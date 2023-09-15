from ...default.encoder import Encoder as encoder_cls
from gym.spaces import Box, Discrete
from light_malib.utils.episode import EpisodeKey
import torch.nn as nn

import numpy as np
from gym.spaces.utils import Box

from pettingzoo.magent import battle_v3
env = battle_v3.parallel_env(map_size=15,max_cycles=500)


class Encoder(encoder_cls):
    def __init__(self):
        super().__init__(action_spaces=env.action_spaces['red_0'],
                         observation_spaces=Box(0.0, 2.0, (845,)), #env.observation_spaces['red_0'],
                         state_space=Box(0.0, 2.0, (845,))) #env.observation_spaces['red_0'])  13*13*5


class Rewarder:
    def __init__(self):
        pass

    def r(self, raw_rewards, **kwargs):

        return np.array([raw_rewards])



class Actor(nn.Module):
    def __init__(
        self,
        model_config,
        observation_space: Box,
        action_space: Discrete,
        custom_config,
        initialization,
    ):
        super().__init__()
        self.attack_dict = {(5, 5): 13, (6, 5): 14, (7, 5): 15, (5, 6): 16, (7, 6): 17, (5, 7): 18, (6, 7): 19,
                       (7, 7): 20}  # coord
        self.attack_coord = list(self.attack_dict.keys())
        self.action_space = env.action_spaces['red_0']

    def forward(self, **kwargs):
        observations = kwargs[EpisodeKey.CUR_OBS]
        oppo_state = observations[:,:,3]
        action_list = []
        for i in self.attack_coord:
            coord1, coord2 = i
            if oppo_state[coord2, coord1] > 0:
                action_list.append(self.attack_dict[i])

        if len(action_list)==0:
            action = np.array(self.action_space.sample())       #move randomly if no opponent is observed
        else:
            action = np.random.choice(action_list, 1)       #random attack an opponent within the view


class Critic(nn.Module):
    def __init__(
        self,
        model_config,
        observation_space: Box,
        action_space: Discrete,
        custom_config,
        initialization,
    ):
        super().__init__()

    def forward(self, **kwargs):
        pass