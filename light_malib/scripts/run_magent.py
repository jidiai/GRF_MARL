from light_malib.rollout.rollout_func_magent import rollout_func

from light_malib.utils.desc.task_desc import RolloutDesc
from light_malib.utils.episode import EpisodeKey
from light_malib.envs.magent.env import Magent
from light_malib.buffer.data_server import DataServer
from light_malib.utils.logger import Logger
from light_malib.utils.naming import default_table_name

import numpy as np
import gym
import time
import math

from dataclasses import dataclass, field
from typing import Any, Dict, List

class MARolloutDesc:
    agent_id: List
    policy_id: Dict
    policy_distributions: Dict
    share_policies: bool
    sync: bool
    stopper: Any
    type: str  # 'rollout', 'evaluation', 'simulation
    kwargs: Dict = field(default_factory=lambda: {})


class DefaultFeatureEncoder:
    def __init__(self, action_spaces, observation_spaces):

        self._action_space = action_spaces
        self._observation_space = observation_spaces

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


class RandomPlayer:
    def __init__(self, action_space, obs_space):
        self.action_space = action_space
        self.feature_encoder = DefaultFeatureEncoder(action_space,
                                                     obs_space)
        self.current_eps = 1

    def get_initial_state(self, batch_size):
        return {
            EpisodeKey.CRITIC_RNN_STATE: np.zeros(1),
            EpisodeKey.ACTOR_RNN_STATE: np.zeros(1),
        }

    def compute_action(self, **kwargs):
        obs = kwargs.get(EpisodeKey.CUR_OBS)
        action = []
        for _ in range(obs.shape[0]):
            action.append(self.action_space.sample())
        action = np.array(action)

        return {
            EpisodeKey.ACTION: action,
            EpisodeKey.CRITIC_RNN_STATE: kwargs[EpisodeKey.CRITIC_RNN_STATE],
            EpisodeKey.ACTOR_RNN_STATE: kwargs[EpisodeKey.ACTOR_RNN_STATE],
        }

def merge_gym_box(box_list):
    length = len(box_list)
    total_shape = box_list[0].shape[0]
    low = box_list[0].low
    high = box_list[0].high
    dtype = box_list[0].dtype

    for i in range(1,length):
        assert box_list[0] == box_list[i], f"box list has unequal elements, {box_list[0] and box_list[i]}"
        low = np.concatenate([low, low])
        high = np.concatenate([high, high])
        total_shape += box_list[i].shape[0]

    return gym.spaces.Box(low=low,high=high, shape=(total_shape,), dtype =dtype)



INDEPENDENT_OBS = False


env_cfg = {'env_id': "battle_v3",
           'map_size': 30, 'max_cycles': 500, 'global_encoder': False}# env_cfg = {'env_id': "simple_speaker_listener_v3", "global_encoder": not INDEPENDENT_OBS}

env = Magent(0, None, env_cfg)
from light_malib.registry.registration import DQN, MAPPO
from light_malib.utils.cfg import load_cfg
# cfg = '/home/yansong/Desktop/jidiai/ai_lib/expr/magent/magent_battle_ppo_marl.yaml'
# cfg = load_cfg(cfg)
# policy_0 = PPO("PPO", env.observation_spaces['red'], env.action_spaces['red'],
#                model_config=cfg['populations'][0]['algorithm']['model_config'],
#                custom_config=cfg['populations'][0]['algorithm']['custom_config'])
cfg = '/home/yansong/Desktop/football_new/GRF_MARL/expr_configs/magent_battle/expr_dqn_marl.yaml'
cfg = load_cfg(cfg)
policy_0 = DQN('DQN', env.observation_spaces['red'], env.action_spaces['red'],
               model_config=cfg['populations'][0]['algorithm']['model_config'],
               custom_config=cfg['populations'][0]['algorithm']['custom_config'])



behavior_policies = {
    "red": ('policy_0', policy_0),
    "blue": ('policy_1', RandomPlayer(env.action_spaces['red'],
                                      env.observation_spaces['red']))
}




rollout_desc = RolloutDesc("agent_0", None, None, None, None, None)


cfg.data_server.table_cfg.rate_limiter_cfg.min_size = 1
datasever = DataServer('dataserver_1', cfg.data_server)
table_name = default_table_name(
    rollout_desc.agent_id,
    rollout_desc.policy_id,
    rollout_desc.share_policies,
)
datasever.create_table(table_name)

from light_malib.registry.registration import DQNTrainer
trainer = DQNTrainer("ppo_trainer")


# print(behavior_policies)
for _ in range(1):
    env = Magent(0, None, env_cfg)

    rollout_results = rollout_func(
        eval=False,
        rollout_worker=None,
        rollout_desc=rollout_desc,
        env=env,
        behavior_policies=behavior_policies,
        data_server=datasever,
        rollout_length=50,
        render=True,
        episode_mode='time-step'
    )

data_list = []
for _ in range(1):
    sample, _ = datasever.sample(table_name, batch_size = 50)
    data_list.append(sample)


def stack(samples):
    ret = {}
    for k, v in samples[0].items():
        # recursively stack
        if isinstance(v, dict):
            ret[k] = stack([sample[k] for sample in samples])
        elif isinstance(v, np.ndarray):
            ret[k] = np.stack([sample[k] for sample in samples])
        elif isinstance(v, list):
            ret[k] = [
                stack([sample[k][i] for sample in samples])
                for i in range(len(v))
            ]
        else:
            raise NotImplementedError
    return ret

#merge data
samples = []
for i in range(len(data_list[0])):
    sample = {}
    for data in data_list:
        sample.update(data[i])
    samples.append(sample)

stack_samples = stack(samples)

policy_0 = policy_0.to_device('cuda:0')


trainer.reset(policy_0, cfg.training_manager.trainer)
for i in range(15):
    training_ret = trainer.optimize(stack_samples)
    Logger.info(f'Training iteration {i}: value loss = {training_ret["value_loss"]}')


Logger.info(f"Training complete")
