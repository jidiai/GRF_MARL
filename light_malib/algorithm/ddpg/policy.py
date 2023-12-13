import copy
import os
import pickle
import random
import gym
import torch
import numpy as np

from torch import nn
from light_malib.utils.logger import Logger
from light_malib.utils.typing import DataTransferType, Tuple, Any, Dict, EpisodeID, List
from light_malib.utils.episode import EpisodeKey
from light_malib.algorithm.common.policy import Policy

import wrapt
import tree
import importlib
from light_malib.utils.logger import Logger
from gym.spaces import Discrete
from ..utils import PopArt
from light_malib.registry import registry
from copy import deepcopy


def hard_update(target, source):
    """Copy network parameters from source to target.

    Reference:
        https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L15

    :param torch.nn.Module target: Net to copy parameters to.
    :param torch.nn.Module source: Net whose parameters to copy
    """

    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


@wrapt.decorator
def shape_adjusting(wrapped, instance, args, kwargs):
    """
    A wrapper that adjust the inputs to corrent shape.
    e.g.
        given inputs with shape (n_rollout_threads, n_agent, ...)
        reshape it to (n_rollout_threads * n_agent, ...)
    """
    offset = len(instance.observation_space.shape)
    original_shape_pre = kwargs[EpisodeKey.CUR_OBS].shape[:-offset]
    num_shape_ahead = len(original_shape_pre)

    def adjust_fn(x):
        if isinstance(x, np.ndarray):
            return np.reshape(x, (-1,) + x.shape[num_shape_ahead:])
        else:
            return x

    def recover_fn(x):
        if isinstance(x, np.ndarray):
            return np.reshape(x, original_shape_pre + x.shape[1:])
        else:
            return x

    adjusted_args = tree.map_structure(adjust_fn, args)
    adjusted_kwargs = tree.map_structure(adjust_fn, kwargs)

    rets = wrapped(*adjusted_args, **adjusted_kwargs)

    recover_rets = tree.map_structure(recover_fn, rets)

    return recover_rets


@registry.registered(registry.POLICY)
class DDPG(Policy):
    def __init__(
            self,
            registered_name: str,
            observation_space: gym.spaces.Space,  # legacy
            action_space: gym.spaces.Space,  # legacy
            model_config: Dict[str, Any] = None,
            custom_config: Dict[str, Any] = None,
            **kwargs,
    ):
        del observation_space
        # del action_space

        self.registered_name = registered_name
        assert self.registered_name == "DDPG"
        self.model_config = model_config
        self.custom_config = custom_config


        model_type = model_config["model"]
        Logger.warning("use model type: {}".format(model_type))
        model = importlib.import_module("light_malib.model.{}".format(model_type))

        FE_cfg = custom_config.get('FE_cfg', None)
        if FE_cfg is not None:
            self.feature_encoder = model.FeatureEncoder(**FE_cfg)
        else:
            self.feature_encoder = model.FeatureEncoder()

        # TODO(jh): extension to multi-agent cooperative case
        # self.env_agent_id = kwargs["env_agent_id"]
        # self.global_observation_space=self.encoder.global_observation_space if hasattr(self.encoder,"global_observation_space") else self.encoder.observation_space
        global_observation_space = self.feature_encoder.global_observation_space
        self.observation_space = self.feature_encoder.observation_space
        self.action_space = self.feature_encoder.action_space

        super(DDPG, self).__init__(
            registered_name=registered_name,
            observation_space=self.observation_space,
            action_space=self.action_space,
            model_config=model_config,
            custom_config=custom_config,
        )

        self.device = torch.device(
            "cuda" if custom_config.get("use_cuda", False) else "cpu"
        )

        self.actor = model.Actor(
            self.model_config["actor"],
            self.observation_space,
            self.action_space,
            self.custom_config,
            self.model_config["initialization"],
        )

        self.critic = model.Critic(
            self.model_config["critic"],
            self.observation_space,
            self.action_space,
            self.custom_config,
            self.model_config["initialization"],
        )

        self.target_critic = deepcopy(self.critic)

        self.share_backbone = False
        # if custom_config["use_popart"]:
        #     self.value_normalizer = PopArt(
        #         1, device=self.device, beta=custom_config["popart_beta"]
        #     )

    @property
    def description(self):
        """Return a dict of basic attributes to identify policy.

        The essential elements of returned description:

        - registered_name: `self.registered_name`
        - observation_space: `self.observation_space`
        - action_space: `self.action_space`
        - model_config: `self.model_config`
        - custom_config: `self.custom_config`

        :return: A dictionary.
        """

        return {
            "registered_name": self.registered_name,
            "observation_space": self.observation_space,
            "action_space": self.action_space,
            "model_config": self.model_config,
            "custom_config": self.custom_config,
        }

    def get_initial_state(self, batch_size):
        return {
            EpisodeKey.ACTOR_RNN_STATE: np.zeros(
                (batch_size, self.actor.rnn_layer_num, self.actor.rnn_state_size)
            ),
            EpisodeKey.CRITIC_RNN_STATE: np.zeros(
                (batch_size, self.critic.rnn_layer_num, self.critic.rnn_state_size)
            ),
        }


    def to_device(self, device):
        self_copy = copy.deepcopy(self)
        self_copy.device = device
        self_copy.actor = self_copy.actor.to(device)
        self_copy.critic = self_copy.critic.to(device)
        self_copy.target_critic = self_copy.target_critic.to(device)

        return self_copy

    @shape_adjusting
    def compute_action(self, **kwargs):
        '''
        TODO(jh): need action sampler, e.g. epsilon-greedy.
        '''
        step = kwargs.get("step", 0)
        to_numpy = kwargs.get("to_numpy", True)
        explore = kwargs["explore"]
        with torch.no_grad():
            obs = kwargs[EpisodeKey.CUR_OBS]
            action_masks = kwargs[EpisodeKey.ACTION_MASK]
            actor_rnn_states = kwargs[EpisodeKey.ACTOR_RNN_STATE]
            critic_rnn_states = kwargs[EpisodeKey.CRITIC_RNN_STATE]
            rnn_masks = kwargs[EpisodeKey.DONE]

            logits, actor_rnn_states, rnn_masks = self.actor(observations=obs,
                                                            actor_rnn_states=actor_rnn_states,
                                                            rnn_masks = rnn_masks,
                                                            action_masks = action_masks)
            dist = torch.distributions.Categorical(logits = logits)
            actions = dist.sample() if explore else dist.probs.argmax(dim=-1)
            dist_entropy = dist.entropy()
            action_log_probs = dist.log_prob(actions)

        if to_numpy:
            actor_rnn_states = actor_rnn_states.detach().cpu().numpy()
            actions = actions.detach().cpu().numpy()
            action_log_probs = action_log_probs.detach().cpu().numpy()
            logits = logits.detach().cpu().numpy()

        ret = {
                EpisodeKey.ACTION: actions,
                EpisodeKey.ACTION_LOG_PROB: logits,
                EpisodeKey.ACTOR_RNN_STATE: actor_rnn_states,
                EpisodeKey.CRITIC_RNN_STATE: critic_rnn_states,
                EpisodeKey.ACTION_ENTROPY: dist_entropy
            }


        return ret

    def eval(self):
        self.actor.eval()
        self.critic.eval()

    def train(self):
        self.actor.train()
        self.critic.train()

    @shape_adjusting
    def value_function(self, **kwargs):
        pass
        # to_numpy = kwargs.get("to_numpy", True)
        # use_target_critic = kwargs.get("use_target_critic", False)
        # if use_target_critic:
        #     critic = self.critic
        # else:
        #     critic = self.target_critic
        # with torch.no_grad():
        #     obs = kwargs[EpisodeKey.CUR_OBS]
        #     action_masks = kwargs[EpisodeKey.ACTION_MASK]
        #     q_values = critic(**{EpisodeKey.CUR_OBS: obs, EpisodeKey.ACTION_MASK: action_masks})
        #     # denormalize
        #     # if hasattr(self,"value_normalizer"):
        #     #     q_values=self.value_normalizer.denormalize(q_values)
        #     if to_numpy:
        #         q_values = q_values.cpu().numpy()
        # return {EpisodeKey.STATE_ACTION_VALUE: q_values,
        #         EpisodeKey.ACTION_MASK: action_masks}

    def dump(self, dump_dir):
        torch.save(self.critic.state_dict(), os.path.join(dump_dir, "critic_state_dict.pt"))
        pickle.dump(self.description, open(os.path.join(dump_dir, "desc.pkl"), "wb"))

    @staticmethod
    def load(dump_dir, **kwargs):
        with open(os.path.join(dump_dir, "desc.pkl"), "rb") as f:
            desc_pkl = pickle.load(f)

        policy = DDPG(
            desc_pkl["registered_name"],
            desc_pkl["observation_space"],
            desc_pkl["action_space"],
            desc_pkl["model_config"],
            desc_pkl["custom_config"],
            **kwargs,
        )

        critic_path = os.path.join(dump_dir, "critic_state_dict.pt")
        if os.path.exists(critic_path):
            critic_state_dict = torch.load(os.path.join(dump_dir, "critic_state_dict.pt"), policy.device)
            policy.critic.load_state_dict(critic_state_dict)
            policy.target_critic = deepcopy(policy.critic)
        return policy