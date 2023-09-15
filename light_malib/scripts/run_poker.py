from light_malib.algorithm.q_learning.policy import QLearning
from light_malib.rollout.rollout_func_aec import rollout_func
from light_malib.utils.desc.task_desc import RolloutDesc
from light_malib.utils.episode import EpisodeKey
from light_malib.envs.kuhn_poker.env import KuhnPokerEnv, DefaultFeatureEncoder
from light_malib.registry import registry
from light_malib.utils.cfg import load_cfg
import numpy as np

from light_malib.buffer.data_server import DataServer
from light_malib.utils.naming import default_table_name
from light_malib.algorithm.q_learning.trainer import QLearningTrainer

from light_malib.registry.registration import MAPPO
from light_malib.registry.registration import MAPPOTrainer

import os
import pathlib
BASE_DIR = str(pathlib.Path(__file__).resolve().parent.parent.parent)


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

class FeatureEncoder:
    def __init__(self):
        pass

    def encode(self, state):
        legal_action_idices = state.legal_actions()
        action_mask = np.zeros(2, dtype=np.float32)
        action_mask[legal_action_idices] = 1
        return state, action_mask


class HumanPlayer:
    def __init__(self):
        self.feature_encoder = FeatureEncoder()

    def get_initial_state(self, batch_size):
        return {
            EpisodeKey.CRITIC_RNN_STATE: np.zeros(1),
            EpisodeKey.ACTOR_RNN_STATE: np.zeros(1)
        }

    def compute_action(self, **kwargs):
        obs = kwargs[EpisodeKey.CUR_OBS][0]
        action_mask = kwargs[EpisodeKey.ACTION_MASK][0]
        valid_actions = np.nonzero(action_mask)[0]
        action = input(
            "player {}: valid actions are {}, please input your action(0-pass,1-bet):".format(obs.current_player(),
                                                                                              valid_actions))
        action = int(action)

        return {
            EpisodeKey.ACTION: action,
            EpisodeKey.CRITIC_RNN_STATE: kwargs[EpisodeKey.CRITIC_RNN_STATE],
            EpisodeKey.ACTOR_RNN_STATE: kwargs[EpisodeKey.ACTOR_RNN_STATE]
        }

cfg_path= 'expr_configs/kuhn_poker/expr_q_learning_psro.yaml' #'expr_configs/poker/expr_q_learning_psro.yaml'
cfg = load_cfg(os.path.join(BASE_DIR, cfg_path))

policy_id_0 = "policy_0"
policy_id_1 = "policy_1"

policy_0 = QLearning('QLearning', None, None, cfg.populations[0].algorithm.model_config,
                     cfg.populations[0].algorithm.custom_config,)
policy_1 = QLearning('QLearning', None, None, cfg.populations[0].algorithm.model_config,
                     cfg.populations[0].algorithm.custom_config,)

trainer = QLearningTrainer('trainer1')

rollout_desc = RolloutDesc("agent_0", None, None, None, None, None)
behavior_policies = {
    "agent_0": (policy_id_0, policy_0),
    "agent_1": (policy_id_1, policy_1)
}

cfg.data_server.table_cfg.rate_limiter_cfg.min_size = 1
datasever = DataServer('dataserver_1', cfg.data_server)
table_name = default_table_name(
    rollout_desc.agent_id,
    rollout_desc.policy_id,
    rollout_desc.share_policies,
)
datasever.create_table(table_name)


for i in range(2):
    env = KuhnPokerEnv(0, None, None)
    results = rollout_func(
        eval=False,
        rollout_worker=None,
        rollout_desc=rollout_desc,
        env=env,
        behavior_policies=behavior_policies,
        data_server=datasever,
        padding_length=10,
        render=True,
        episode_mode=cfg.rollout_manager.worker.episode_mode
    )

data_list = []
for _ in range(1):
    sample, _ = datasever.sample(table_name, batch_size = 2)
    data_list.append(sample)


#merge data
samples = []
for i in range(len(data_list[0])):
    sample = {}
    for data in data_list:
        sample.update(data[i])
    samples.append(sample)

stack_samples = stack(samples)

class trainer_cfg:
    policy = policy_0

policy_0 = policy_0.to_device('cuda:0')
trainer.reset(policy_0, cfg.training_manager.trainer)
training_results = trainer.optimize(stack_samples)

print(results)