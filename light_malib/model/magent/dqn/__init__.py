from ...default.dqn_actor import Actor
from ...default.encoder import Encoder as encoder_cls
from gym.spaces import Box, Discrete
import numpy as np
from light_malib.utils.episode import EpisodeKey
import torch
import torch.nn as nn

import numpy as np
from gym.spaces.utils import Box

def to_tensor(arr):
    if isinstance(arr, np.ndarray):
        arr = torch.FloatTensor(arr)
    return arr


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
        # game = pyspiel.load_game('kuhn_poker')
        # policy=policy_lib.TabularPolicy(game)
        # assert action_space.n==policy.action_probability_array.shape[-1]
        # self.q_table=nn.Parameter(torch.zeros(size=policy.action_probability_array.shape))
        # torch.nn.init.uniform_(self.q_table,a=0,b=0.001)
        in_dim = observation_space.shape[0]
        out_dim = action_space.n if isinstance(action_space, Discrete) else action_space.shape[0]
        hidden_size = 64

        # self.q_net = nn.Sequential(
        #     nn.Linear(in_dim, hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size, hidden_size),
        #     nn.ReLU(),
        #     nn.Linear(hidden_size, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, out_dim)
        # )
        self.q_net = nn.Sequential(
            nn.Linear(in_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size,hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_dim)
        )

        self._init()

    def _init(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                nn.init.uniform_(module.bias, 0, 0.001)
            else:
                pass

    def forward(self, **kwargs):
        observations = to_tensor(kwargs[EpisodeKey.CUR_OBS])
        action_masks = to_tensor(kwargs[EpisodeKey.ACTION_MASK])
        # print(observations.shape,action_masks.shape)

        # observations are encoded as index to q_table
        assert len(observations.shape) == 2,print(f'critic obs shape = {observations.shape}')
        q_values = self.q_net(observations)
        # mask out invalid actions
        if not action_masks.shape == q_values.shape:
            breakpoint()
        q_values = action_masks * q_values + (1 - action_masks) * (-10e9)
        return q_values
