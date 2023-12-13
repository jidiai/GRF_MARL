from light_malib.algorithm.common.rnn_net import RNNNet
from gym.spaces.box import Box
import numpy as np

class Critic(RNNNet):
    def __init__(
        self,
        model_config,
        observation_space,
        action_space,
        custom_config,
        initialization,
    ):

        observation_shape = observation_space.shape
        action_num = action_space.n
        input_shape = (observation_shape[0]+action_num, )
        output_shape = (1,)
        input_space = Box(low=np.ones(input_shape[0])*-1000,
                          high=np.ones(input_shape[0])*1000,
                           shape=input_shape,
                          dtype=observation_space.dtype)
        output_shape = Box(low=np.ones(1)*-1000,
                          high=np.ones(1)*1000,
                           shape=output_shape,
                           dtype=observation_space.dtype)

        super().__init__(
            model_config, input_space, output_shape, custom_config, initialization
        )