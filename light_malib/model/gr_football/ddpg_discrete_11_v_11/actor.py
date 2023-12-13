from light_malib.utils.episode import EpisodeKey

import torch
import torch.nn as nn
import numpy as np
from light_malib.algorithm.common.rnn_net import RNNNet


def to_tensor(arr):
    if isinstance(arr, np.ndarray):
        arr = torch.FloatTensor(arr)
    return arr



class Actor(RNNNet):
    def __init__(
            self,
            model_config,
            observation_space,
            action_space,
            custom_config,
            initialization,
    ):
        super().__init__(
            model_config, observation_space, action_space, custom_config, initialization
        )

    def forward(self, observations, actor_rnn_states, rnn_masks, action_masks):
        logits, actor_rnn_states = super().forward(
            observations, actor_rnn_states, rnn_masks
        )
        illegal_action_mask = 1 - action_masks
        logits = logits - 1e10 * illegal_action_mask

        return logits, actor_rnn_states, rnn_masks


