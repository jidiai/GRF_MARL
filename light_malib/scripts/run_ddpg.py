# Copyright 2022 Digital Brain Laboratory, Yan Song and He jiang
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from light_malib.rollout.rollout_func import rollout_func
from light_malib.utils.desc.task_desc import RolloutDesc
from light_malib.utils.episode import EpisodeKey
from light_malib.envs.gr_football.env import GRFootballEnv
from light_malib.algorithm.mappo.policy import MAPPO
from light_malib.utils.cfg import load_cfg
from light_malib.utils.logger import Logger
import numpy as np
import pickle as pkl
import argparse
import copy

from light_malib.buffer.data_server import DataServer
from light_malib.utils.naming import default_table_name

import os
import pathlib
BASE_DIR = str(pathlib.Path(__file__).resolve().parent.parent.parent)


parser = argparse.ArgumentParser(
    description="play google research football competition"
)
parser.add_argument(
    "--config", type=str, default="expr_configs/cooperative_MARL_benchmark/full_game/11_vs_11_hard/ddpg.yaml"
)
# parser.add_argument(
#     "--model_0",
#     type=str,
#     default="light_malib/trained_models/gr_football/11_vs_11/beat_bot",
# )
parser.add_argument(
    "--model_1",
    type=str,
    default="light_malib/trained_models/gr_football/11_vs_11/built_in",
)
parser.add_argument("--render", default=False, action="store_true")
parser.add_argument("--total_run", default=1, type=int)
args = parser.parse_args()

config_path = args.config
# model_path_0 = os.path.join(BASE_DIR, args.model_0)
model_path_1 = os.path.join(BASE_DIR, args.model_1)

cfg = load_cfg(os.path.join(BASE_DIR, config_path))
cfg["rollout_manager"]["worker"]["envs"][0]["scenario_config"]["render"] = args.render

policy_id_0 = "policy_0"
policy_id_1 = "policy_1"

# from light_malib.registry.registration import QMix
# policy_0 = QMix('QMix', None, None, cfg.populations[0].algorithm.model_config, cfg.populations[0].algorithm.custom_config)
from light_malib.registry.registration import QMix, MAPPO, BC, DDPG
# policy_0 = QMix('QMix', None, None, cfg.populations[0].algorithm.model_config, cfg.populations[0].algorithm.custom_config)
# policy_0 = MAPPO('MAPPO', None, None, cfg.populations[0].algorithm.model_config, cfg.populations[0].algorithm.custom_config,
#                  env_agent_id='agent_0')
# policy_0 = BC("BC", None, None, cfg.populations[0].algorithm.model_config, cfg.populations[0].algorithm.custom_config)
policy_0 = DDPG("DDPG", None, None, cfg.populations[0].algorithm.model_config, cfg.populations[0].algorithm.custom_config)

# policy_0 = MAPPO.load(model_path_0, env_agent_id="agent_0")
policy_1 = MAPPO.load(model_path_1, env_agent_id="agent_1")

env = GRFootballEnv(0, None, cfg.rollout_manager.worker.envs[0])
rollout_desc = RolloutDesc("agent_0", None, None, None, None, None)

cfg.data_server.table_cfg.rate_limiter_cfg.min_size = 1
datasever = DataServer('dataserver_1', cfg.data_server)
table_name = default_table_name(
    rollout_desc.agent_id,
    rollout_desc.policy_id,
    rollout_desc.share_policies,
)
datasever.create_table(table_name)

# from light_malib.algorithm.qmix.trainer import QMixTrainer
# trainer = QMixTrainer('trainer_1')
# from light_malib.algorithm.mappo.trainer import MAPPOTrainer
# trainer = MAPPOTrainer('trainer_1')
# from light_malib.registry.registration import BCTrainer
# trainer = BCTrainer('trainer_1')
from light_malib.algorithm.ddpg.trainer import DDPGTrainer
trainer = DDPGTrainer('trainer_1')

total_run = args.total_run
total_win = 0
offset = np.random.randint(0, 2)
for idx in range(1):
    env = GRFootballEnv(0, None, cfg.rollout_manager.worker.envs[0])
    rollout_desc = RolloutDesc("agent_0", None, None, None, None, None)

    # if (offset + idx) % 2 == 0:
    agent = "agent_0"
    behavior_policies = {
        "agent_0": (policy_id_0, policy_0),
        "agent_1": (policy_id_1, policy_1),
    }
    Logger.info("run {}/{}: model_0 vs model_1".format(idx + 1, total_run))
    # else:
    #     agent = "agent_1"
    #     behavior_policies = {
    #         "agent_0": (policy_id_1, policy_1),
    #         "agent_1": (policy_id_0, policy_0),
    #     }
    #     Logger.info("run {}/{}: model_1 vs model_0".format(idx + 1, total_run))
    rollout_results = rollout_func(
        eval=False,
        rollout_worker=None,
        rollout_desc=rollout_desc,
        env=env,
        behavior_policies=behavior_policies,
        data_server=datasever,
        rollout_length=500, #cfg.rollout_manager.worker.rollout_length,
        sample_length=0, #cfg.rollout_manager.worker.sample_length,
        render=False,
        rollout_epoch=100,
        episode_mode=cfg.rollout_manager.worker.episode_mode,
    )
    Logger.info("stats of model_0 is {}".format(rollout_results['results'][0]["stats"][agent]))
    # total_win += rollout_results["stats"][agent]["win"]
Logger.warning("win rate of model_0 is {}".format(total_win / total_run))

data_list = []
for _ in range(1):
    sample, _ = datasever.sample(table_name, batch_size = 5)
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

for i, j in stack_samples.items():
    stack_samples[i] = np.repeat(j, [10], axis=0)


class trainer_cfg:
    policy = policy_0

# from light_malib.training.distributed_trainer import DistributedTrainer
# from light_malib.utils.naming import default_trainer_id
# dis_trainer = [DistributedTrainer(id=default_trainer_id(idx),
#                                   local_rank=idx,
#                                   world_size=cfg.training_manager.num_trainers,
#                                   master_addr=cfg.training_manager.master_addr,
#                                   master_posrt=cfg.training_manager.master_port,
#                                   master_ifname=cfg.training_manger.get("master_ifname", None),
#                                   gpu_preload=cfg.training_manger.gpu_preload,
#                                   local_queue_size=cfg.training_manager.local_queue_size,
#                                   policy_server)]


policy_0 = policy_0.to_device('cuda:0')

trainer.reset(policy_0, cfg.training_manager.trainer)
trainer.optimize(stack_samples)

Logger.info(f"Training complete")



