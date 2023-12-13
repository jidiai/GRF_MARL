# -*- coding: utf-8 -*-
from tkinter import TRUE
import torch
import torch.nn.functional as F
from light_malib.utils.episode import EpisodeKey
from light_malib.algorithm.common.loss_func import LossFunc
from light_malib.utils.logger import Logger
from light_malib.registry import registry

def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (e > d).float()
    return a * e ** 2 / 2 + b * d * (abs(e) - d / 2)

def mse_loss(e):
    return (e ** 2) / 2

def to_value(tensor:torch.Tensor):
    return tensor.detach().cpu().item()

def basic_stats(name,tensor:torch.Tensor):
    stats={}
    stats["{}_max".format(name)]=to_value(tensor.max())
    stats["{}_min".format(name)]=to_value(tensor.min())
    stats["{}_mean".format(name)]=to_value(tensor.mean())
    stats["{}_std".format(name)]=to_value(tensor.std())
    return stats

def soft_update(target, source, tau):
    """Perform DDPG soft update (move target params toward source based on weight factor tau).

    Reference:
        https://github.com/ikostrikov/pytorch-ddpg-naf/blob/master/ddpg.py#L11

    :param torch.nn.Module target: Net to copy parameters to
    :param torch.nn.Module source: Net whose parameters to copy
    :param float tau: Range form 0 to 1, weight factor for update
    """

    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

@registry.registered(registry.LOSS)
class DDPGLoss(LossFunc):
    def __init__(self):
        # TODO: set these values using custom_config
        super().__init__()

        self._use_huber_loss = False
        if self._use_huber_loss:
            self.huber_delta = 10.0

        self._use_max_grad_norm = True
        

    def reset(self, policy, config):
        """Replace critic with a centralized critic"""
        self._params.update(config)
        if policy is not self.policy:
            self._policy = policy
            # self._set_centralized_critic()
            self.setup_optimizers()
        self.step_ctr=0
        self.clip_param = policy.custom_config.get("clip_param", 0.2)
        self.max_grad_norm = policy.custom_config.get("max_grad_norm", 10)

    def setup_optimizers(self, *args, **kwargs):
        """Accept training configuration and setup optimizers"""

        optim_cls = getattr(torch.optim, self._params.get("optimizer", "Adam"))

        param_groups = []

        if len(list(self._policy.actor.parameters())) > 0:
            param_groups.append({'params': self.policy.actor.parameters(), 'lr': self._params["actor_lr"]})

        if len(list(self._policy.critic.parameters())) > 0:
            param_groups.append({'params': self.policy.critic.parameters(), 'lr': self._params["critic_lr"]})

        self.optimizer=optim_cls(
            param_groups,
            eps=self._params["opti_eps"],
            weight_decay=self._params["weight_decay"]
        )
        self.optimizer.zero_grad()

        self.n_opt_steps = 0


    def loss_compute(self, sample):
        self.step_ctr+=1
        policy=self._policy
        self.max_grad_norm = policy.custom_config.get("max_grad_norm",10)
        self.gamma = policy.custom_config["gamma"]
        target_update_freq=policy.custom_config["target_update_freq"]
        target_update_lr=policy.custom_config["target_update_lr"]

        (
            observations,
            action_masks,
            actions,
            actions_logits,
            rewards,
            dones,
            next_observations,
            next_action_masks
        ) = (
            sample[EpisodeKey.CUR_OBS],
            sample[EpisodeKey.ACTION_MASK],
            sample[EpisodeKey.ACTION].long(),
            sample[EpisodeKey.ACTION_LOG_PROB],
            sample[EpisodeKey.REWARD],
            sample[EpisodeKey.DONE],
            sample[EpisodeKey.NEXT_OBS],
            sample[EpisodeKey.NEXT_ACTION_MASK]
        )

        self.optimizer.zero_grad()
        # actor loss
        current_action_logits, _, _ = self.policy.actor(observations=observations,
                                                  actor_rnn_states=torch.ones(1),
                                                  rnn_masks=dones,
                                                  action_masks=action_masks)

        differentiable_a = torch.nn.functional.gumbel_softmax(
            logits=current_action_logits
        )
        obs_a = torch.concatenate([observations, differentiable_a], dim=-1)
        actor_loss, _ = self.policy.critic(obs_a, rnn_states=torch.ones(1), masks=dones)
        actor_loss = -actor_loss.mean()


        with torch.no_grad():
            next_action_logits, _, _ = self.policy.actor(observations=next_observations,
                                                   actor_rnn_states = torch.ones(1),
                                                   rnn_masks=dones,
                                                   action_masks=next_action_masks)
            next_action = torch.nn.functional.softmax(next_action_logits)
            target_obs_action = torch.concatenate([next_observations, next_action], dim=-1)
            next_value, _ = self.policy.target_critic(target_obs_action,
                                                   rnn_states=torch.ones(1), masks=dones)
            target_value = rewards + self.gamma*next_value*(1.0-dones)

        actions_logprob = torch.nn.functional.softmax(actions_logits)
        true_obs_action = torch.concatenate([observations, actions_logprob], dim=-1)
        true_value, _= self.policy.critic(true_obs_action,
                                        rnn_states=torch.ones(1), masks=dones)
        assert true_value.shape==target_value.shape, print(f"True value shape = {true_value.shape}"
                                                           f"Target value shape = {target_value.shape}")
        value_loss = torch.nn.MSELoss()(true_value, target_value.detach())

        total_loss = actor_loss + value_loss
        total_loss.backward()
        if self._use_max_grad_norm:
            for param_group in self.optimizer.param_groups:
                torch.nn.utils.clip_grad_norm_(
                    param_group["params"], self.max_grad_norm
                )
        self.optimizer.step()

        if self.step_ctr%target_update_freq==0:
            soft_update(self.policy.target_critic, self.policy.critic, target_update_lr)

        stats={
            "value_loss":float(value_loss.detach().cpu().numpy())
        }
        return stats

    def zero_grad(self):
        pass

    def step(self):
        pass
