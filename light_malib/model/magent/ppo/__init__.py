from ...default.ppo_actor import Actor
from ...default.ppo_critic import Critic
from ...default.encoder import Encoder as encoder_cls

import numpy as np
from gym.spaces.utils import Box

from pettingzoo.magent import battle_v3
env = battle_v3.parallel_env(map_size=15,max_cycles=500)

class Encoder(encoder_cls):
    def __init__(self):
        super().__init__(action_spaces=env.action_spaces['red_0'],
                         observation_spaces=Box(0.0, 2.0, (854,)), #env.observation_spaces['red_0'],
                         state_space=Box(0.0, 2.0, (854,))) #env.observation_spaces['red_0'])


class Rewarder:
    def __init__(self):
        pass

    def r(self, raw_rewards, **kwargs):

        return np.array([raw_rewards])
