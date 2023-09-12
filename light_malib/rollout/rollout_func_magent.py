from ensurepip import bootstrap
from typing import OrderedDict
import numpy as np
from light_malib.utils.episode import EpisodeKey
from light_malib.envs.base_env import BaseEnv
from light_malib.utils.desc.task_desc import RolloutDesc
from light_malib.utils.timer import global_timer
from light_malib.utils.naming import default_table_name

import time

def rename_field(data, field, new_field):
    for agent_id, agent_data in data.items():
        field_data = agent_data.pop(field)
        agent_data[new_field] = field_data
    return data


def select_fields(data, fields):
    rets = {
        agent_id: {field: agent_data[field] for field in fields if field in agent_data}
        for agent_id, agent_data in data.items()
    }
    return rets


def update_fields(data1, data2):
    def update_dict(dict1, dict2):
        d = {}
        d.update(dict1)
        d.update(dict2)
        return d

    rets = {
        agent_id: update_dict(data1[agent_id], data2[agent_id]) for agent_id in data1
    }
    return rets


def stack_step_data(step_data_list, bootstrap_data):
    episode_data = {}
    for field in step_data_list[0]:
        data_list = [step_data[field] for step_data in step_data_list]
        if field in bootstrap_data:
            data_list.append(bootstrap_data[field])
        episode_data[field] = np.stack(data_list)
    return episode_data


def merge_rets(rets):
    aid_list = list(rets.keys())
    merged_rets={}
    # merged_rets = {aid_list[0]: rets[aid_list[0]]}
    # for aid in aid_list[1:]:
    #     for tag in merged_rets[aid_list[0]]:
    #         tag = np.stack()
    for tag in rets[aid_list[0]]:
        merged_rets[tag] = np.stack([rets[aid][tag] for aid in aid_list])

    return {aid_list[0]: merged_rets}




def rollout_func(
    eval: bool,
    rollout_worker,
    rollout_desc: RolloutDesc,
    env: BaseEnv,
    behavior_policies,
    data_server,
    rollout_length,
    **kwargs
):
    """
    TODO(jh): modify document

    Rollout in simultaneous mode, support environment vectorization.

    :param VectorEnv env: The environment instance.
    :param Dict[Agent,AgentInterface] agent_interfaces: The dict of agent interfaces for interacting with environment.
    :param ray.ObjectRef dataset_server: The offline dataset server handler, buffering data if it is not None.
    :return: A dict of rollout information.
    """

    sample_length = kwargs.get("sample_length", rollout_length)
    render = kwargs.get("render", False)
    episode_mode = kwargs['episode_mode']
    agent_id_list = rollout_desc.agent_id

    policy_ids = OrderedDict()
    feature_encoders = OrderedDict()
    for agent_id, (policy_id, policy) in behavior_policies.items():
        feature_encoders[agent_id] = policy.feature_encoder
        policy_ids[agent_id] = policy_id

    custom_reset_config = {
        "feature_encoders": feature_encoders,
        "main_agent_id": rollout_desc.agent_id,
        "rollout_length": rollout_length,
    }
    # {agent_id:{field:value}}
    share_policies = rollout_desc.share_policies
    global_timer.record("env_step_start")
    env_rets = env.reset(custom_reset_config)

    if render:
        env.render()

    global_timer.time("env_step_start", "env_step_end", "env_step")

    init_rnn_states = {
        agent_id: behavior_policies[agent_id][1].get_initial_state(
            batch_size=env.num_players[agent_id]
        )
        for agent_id in env.agent_ids
    }

    # TODO(jh): support multi-dimensional batched data based on dict & list using sth like NamedIndex.
    step_data = update_fields(env_rets, init_rnn_states)

    step = 0
    step_data_list = []
    while (
        not env.is_terminated()
    ):  # XXX(yan): terminate only when step_length >= fragment_length
        # prepare policy input
        policy_inputs = step_data
        policy_outputs = {}
        global_timer.record("inference_start")
        for agent_id, (policy_id, policy) in behavior_policies.items():
            policy_outputs[agent_id] = policy.compute_action(**policy_inputs[agent_id],
                                                             explore = not eval)
        global_timer.time("inference_start", "inference_end", "inference")

        actions = select_fields(policy_outputs, [EpisodeKey.ACTION])

        global_timer.record("env_step_start")
        env_rets = env.step(actions)
        global_timer.time("env_step_start", "env_step_end", "env_step")

        if render:
            env.render()
            # time.sleep(0.05)

        step_data = update_fields(step_data, select_fields(env_rets,
                                                           [
                                                               EpisodeKey.NEXT_OBS,
                                                               EpisodeKey.NEXT_STATE,
                                                               EpisodeKey.REWARD,
                                                               EpisodeKey.DONE,
                                                               EpisodeKey.NEXT_ACTION_MASK
                                                           ]))


        step_data = update_fields(
            step_data,
            select_fields(
                policy_outputs,
                [EpisodeKey.ACTION, EpisodeKey.ACTION_DIST, EpisodeKey.STATE_VALUE],
            ),
        )

        # save data of trained agent for training

        step_data_list.append(step_data[rollout_desc.agent_id[0]])

        env_rets=rename_field(env_rets, EpisodeKey.NEXT_OBS, EpisodeKey.CUR_OBS)
        env_rets=rename_field(env_rets, EpisodeKey.NEXT_STATE, EpisodeKey.CUR_STATE)
        env_rets=rename_field(env_rets, EpisodeKey.NEXT_ACTION_MASK, EpisodeKey.ACTION_MASK)

        # record data for next step
        step_data = update_fields(
            env_rets,
            select_fields(
                policy_outputs,
                [
                 EpisodeKey.CUR_OBS,
                 EpisodeKey.CUR_STATE,
                 EpisodeKey.ACTION_MASK,
                 EpisodeKey.ACTOR_RNN_STATE,
                 EpisodeKey.CRITIC_RNN_STATE],
            ),
        )

        step += 1

    if not eval:
        if episode_mode == 'traj':
            episode_dict = {aid: [stack_step_data(step_data_list[aid], {})] for aid in agent_id_list}
        elif episode_mode == 'time-step':
            episode_dict = step_data_list
        else:
            raise NotImplementedError(f'Episode mode {episode_mode} not implemented')

        for aid in agent_id_list:
            # breakpoint()
            # tabklenale!!!!
            table_name = f'{aid}_{rollout_desc.policy_id[aid][0]}'
            data_server.save(
                table_name
                ,
                episode_dict,
            )

    stats = env.get_episode_stats()

    # if not eval:
    #     stats['agent_0']['eps'] = current_eps
    return {
        "main_agent_id": rollout_desc.agent_id,
        "policy_ids": policy_ids,
        "stats": stats,
    }
