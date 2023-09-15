import importlib
import gym
from light_malib.registry import registry
from light_malib.utils.episode import EpisodeKey
from light_malib.utils.logger import Logger
from ..base_env import BaseEnv
from light_malib.utils.episode import EpisodeKey
import numpy as np
import time

class DefaultFeatureEncoder:
    def __init__(self, action_spaces, observation_spaces, current_aid):

        self._action_space = action_spaces
        self._observation_space = observation_spaces
        self.current_aid = current_aid

    def encode(self, state):        #flatten the obs
        # obs=np.array([self._policy.state_index(state)],dtype=int)
        # print(self._policy.state_index(state))
        obs = state[self.current_aid]
        n_player,dim1, dim2, dim3 = obs.shape
        obs = obs.reshape(n_player, -1)
        action_mask = np.ones((n_player,self._action_space.n), dtype=np.float32)
        return obs, action_mask

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

class IndividualFeatureEncoder:
    def __init__(self, action_spaces, observation_spaces, current_aid):

        self._action_space = action_spaces
        self._observation_space = observation_spaces
        self.current_aid = current_aid

    def encode(self, state):
        # obs=np.array([self._policy.state_index(state)],dtype=int)
        # print(self._policy.state_index(state))
        obs = state[self.current_aid]
        action_mask = np.ones(self._action_space.n, dtype=np.float32)
        return obs, action_mask

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space




@registry.registered(registry.ENV, "magent")
class Magent(BaseEnv):
    def __init__(self, id, seed, cfg):
        super().__init__(id, seed)
        self.id = id
        self.seed = seed
        self.cfg = cfg
        env_id = cfg["env_id"]
        env_module = importlib.import_module(f"pettingzoo.magent.{env_id}")
        map_size = cfg['map_size']
        max_cycles = cfg['max_cycles']
        # shuffle_init = cfg['shuffle_init']

        self._env = env_module.parallel_env(map_size=map_size, max_cycles=max_cycles)            #max_cycles=25, continuous_actions=False
        self._env.seed(seed)      #Calling seed externally is deprecated in new pettingzoo version
        self._step_ctr = 0
        self._is_terminated = False
        self.rollout_length = max_cycles
        self.env_handle = self._env.env.get_handles()       #handles align with possible agent: [red, blue]

        all_agent_ids = self._env.possible_agents
        team_ids = [i.split('_')[0] for i in all_agent_ids]
        self.agent_ids = ['red', 'blue']       #list(set(team_ids))
        self.all_agent_ids = all_agent_ids

        self.mapped_agent_ids = [f"agent_{i}" for i in range(len(self.agent_ids))]
        self.agent_id_mapping = dict(zip(self.agent_ids, self.mapped_agent_ids))
        self.agent_id_reverse_mapping = dict(zip(self.mapped_agent_ids,self.agent_ids))

        self.env_handle_dict = dict(zip(self.agent_ids, self.env_handle))

        self.num_players = {}
        for aid in self.agent_ids:
            num = team_ids.count(aid)
            self.num_players[aid] = num

        self._observation_space = {aid: self._env.observation_spaces[f'{aid}_0']
                              for aid in self.agent_ids}
        self._action_space = {aid: self._env.action_spaces[f'{aid}_0']
                              for aid in self.agent_ids}


        _global_encoder = self.cfg['global_encoder']
        if _global_encoder is True:
            global_encoder = self.agent_ids
        elif _global_encoder is False:
            global_encoder = []
        elif isinstance(_global_encoder, str):
            global_encoder = _global_encoder.split(',')
        else:
            raise NotImplementedError

        self.feature_encoders = {}
        for aid in self.agent_ids:
            # if aid in global_encoder:
            self.feature_encoders[aid] = DefaultFeatureEncoder(self._action_space[aid],
                                                                   self._observation_space[aid],aid)
        # self.stats = {self.agent_id_mapping[agent_id]: {
        #     "score": 0.0,
        #     "total_reward": 0.0,
        #     f"{agent_id}'s reward": 0,
        # } for agent_id in self.agent_ids
        # }

            # else:
            #     self.feature_encoders[aid] = IndividualFeatureEncoder(self._action_space[aid], self._observation_space[aid], aid)
    @property
    def possible_agents(self):
        return self.agent_ids

    @property
    def action_spaces(self):
        return self._action_space

    @property
    def observation_spaces(self):
        return self._observation_space

    def parameter_sharing_merge(self, state):

        merged_obs = {aid: [] for aid in self.agent_ids}
        for pid, v in state.items():
            team_id = pid.split('_')[0]
            if isinstance(v, float) or isinstance(v, int):
                merged_obs[team_id].append(np.array([v]))
            else:
                merged_obs[team_id].append(v)

        for tid, merged_v in merged_obs.items():
            merged_obs[tid] = np.stack(merged_v)

        return merged_obs


    def reset(self, custom_reset_config=None):
        self._step_ctr = 0
        self._is_terminated=False

        # self.stats={
        #     "total_steps": 0,
        # }

        total_alive = sum([sum(self._env.env.get_alive(i)) for i in self.env_handle])

        self.stats={self.agent_id_mapping[agent_id]: {
            "alive agents":sum(self._env.env.get_alive(self.env_handle_dict[agent_id])),
            "oppo alive agents": total_alive-sum(self._env.env.get_alive(self.env_handle_dict[agent_id])),
            "score": 0.0,
            "reward": 0,
            "win": 0.,
            "total_steps": 0,
        } for agent_id in self.agent_ids
        }

        # for aid in self.agent_ids:
        #     self.stats[aid][f"{aid}'s reward"] = 0

        all_obs = self._env.reset()
        observations = self.parameter_sharing_merge(all_obs)

        encoded_obs = {}
        action_masks = {}
        dones = {}
        for agent_id in self.agent_ids:
            _obs, _action_mask = self.feature_encoders[agent_id].encode(observations)
            encoded_obs[agent_id] = np.array(_obs, dtype=np.float32)
            action_masks[agent_id] = np.array(_action_mask)
            dones[agent_id] = np.zeros((_action_mask.shape[0], 1))

        rets = {
            self.agent_id_mapping[agent_id]:{
                EpisodeKey.CUR_OBS: encoded_obs[agent_id],
                EpisodeKey.CUR_STATE: encoded_obs[agent_id],
                EpisodeKey.ACTION_MASK: action_masks[agent_id],
                EpisodeKey.DONE: dones[agent_id]
            }
            for agent_id in self.agent_ids
        }

        return rets

    def step(self, actions):
        self._step_ctr+=1

        filtered_actions = {}
        for aid, action_set in actions.items():
            pid =0
            if len(action_set[EpisodeKey.ACTION].shape)==3:
                action_set[EpisodeKey.ACTION] = action_set[EpisodeKey.ACTION][0]
            for a in action_set[EpisodeKey.ACTION]:
                filtered_actions[f'{self.agent_id_reverse_mapping[aid]}_{pid}'] = a
                pid+=1


        # filtered_actions = {
        #     aid: int(actions[aid]) for aid in self.agent_ids
        # }

        all_observations, all_rewards, all_dones, infos = self._env.step(filtered_actions)
        # if len(all_observations)!=6:
        #     print('1')


        observations = self.parameter_sharing_merge(all_observations)

        rewards = self.parameter_sharing_merge(all_rewards)
        dones = self.parameter_sharing_merge(all_dones)

        encoded_obs = {}
        action_masks = {}
        for agent_id in self.agent_ids:
            _obs, _action_masks = self.feature_encoders[agent_id].encode(observations)
            encoded_obs[agent_id] = np.array(_obs, dtype=np.float32)
            action_masks[agent_id] = np.array(_action_masks)

        self._is_terminated = all(list(all_dones.values()))
        self.update_episode_stats(rewards)

        for i, agent_id in enumerate(self.agent_ids):
            if encoded_obs[agent_id].shape[0] < self.num_players[agent_id]:             #padding
                add_on_num = self.num_players[agent_id]-encoded_obs[agent_id].shape[0]
                encoded_obs[agent_id] = np.concatenate([encoded_obs[agent_id],
                                                        np.zeros((add_on_num,
                                                                  encoded_obs[agent_id].shape[-1]))])
                action_masks[agent_id] = np.concatenate([action_masks[agent_id],
                                                         np.ones((add_on_num,
                                                                   action_masks[agent_id].shape[-1]))])
                rewards[agent_id] = np.concatenate([rewards[agent_id],
                                                    np.zeros((add_on_num,
                                                              rewards[agent_id].shape[-1]))])

                dones[agent_id] = np.concatenate([dones[agent_id],
                                                  np.zeros((add_on_num,
                                                            dones[agent_id].shape[-1]))])
        return {
            self.agent_id_mapping[agent_id]: {
                EpisodeKey.NEXT_OBS: encoded_obs[agent_id],
                EpisodeKey.NEXT_STATE: encoded_obs[agent_id],
                EpisodeKey.NEXT_ACTION_MASK: action_masks[agent_id],
                EpisodeKey.REWARD: np.array(rewards[agent_id]),
                EpisodeKey.DONE: np.array(dones[agent_id])
            }   for agent_id in self.agent_ids
        }

    def render(self, *args, **kwargs):
        self._env.render()
        time.sleep(0.03)

    def update_episode_stats(self, reward):

        total_alive = sum([sum(self._env.env.get_alive(i)) for i in self.env_handle])
        for agent_id, r in reward.items():
            self.stats[self.agent_id_mapping[agent_id]]["reward"] += sum(r)
            _alive = sum(self._env.env.get_alive(self.env_handle_dict[agent_id]))
            self.stats[self.agent_id_mapping[agent_id]]['alive agents'] = sum(self._env.env.get_alive(self.env_handle_dict[agent_id]))
            self.stats[self.agent_id_mapping[agent_id]]['oppo alive agents'] = total_alive - _alive
            self.stats[self.agent_id_mapping[agent_id]]['total_steps'] = self._step_ctr

            if self._is_terminated:
                if total_alive - _alive == 0 and _alive >0:
                    #all opponents are dead and ours are not
                    self.stats[self.agent_id_mapping[agent_id]]['win'] = 1.

                if _alive > total_alive - _alive:       #we have more soldiers left
                    self.stats[self.agent_id_mapping[agent_id]]['score'] = 1.
                elif _alive < total_alive - _alive:
                    self.stats[self.agent_id_mapping[agent_id]]['score'] = 0.
                else:
                    self.stats[self.agent_id_mapping[agent_id]]['score'] = 0.5

    def get_episode_stats(self):
        return self.stats

    def is_terminated(self):
        return self._is_terminated
    def close(self):
        pass

