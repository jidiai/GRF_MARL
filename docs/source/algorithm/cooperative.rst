Cooperative Learning --- MARL
======================================================================

.. contents::
    :local:
    :depth: 2

----------------------

In each scenario against a fixed opponent, we want to perform cooperative learning to win the game. This is done by computing a MARL solution using
the following approaches.



Independent Learning
---------------------------------------------------------


Independent Proximal Policy Optimization (IPPO)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

IPPO is a fully decentralized variant of the
PPO  algorithm, where each agent is trained independently using its local information. General training
techniques such as General Advantage Estimate (GAE) and value target clipping are employed.

Example:

.. code-block:: shell

    python3 light_malib/main_pb.py --config expr_configs/cooperative_MARL_benchmark/academy/3_vs_1_with_keeper/ippo.yaml



Centralized Training Decentralized Execution (CTDE)
-----------------------------------------------------------


Multi-Agent Proximal Policy Optimization (MAPPO)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

MAPPO follows the Centralized Training Decentralized Execution (CTDE) paradigm, enabling
actors to access local information while critics exploit global state information. Critics with more comprehensive
information typically yield better value estimation, thus resulting in improved policies.

Example:

.. code-block:: shell

    python3 light_malib/main_pb.py --config expr_configs/cooperative_MARL_benchmark/academy/3_vs_1_with_keeper/mappo.yaml


Heterogeneous-Agent PPO (HAPPO)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

HAPPO and A2PO update agents sequentially to prevent the potential conflicting direction updates that may happen in the
simultaneous updating. HAPPO leverages the multi-agent advantage decomposition lemma and guarantees
a monotonic improvement on the joint policy.

Example:

.. code-block:: shell

    python3 light_malib/main_pb.py --config expr_configs/cooperative_MARL_benchmark/academy/3_vs_1_with_keeper/happo.yaml


Agent-by-agent Policy Optimization (A2PO)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A2PO derives an updating formula similar to PPO for each
agent and resolves the preceding agent policy shifting issue using a technique called preceding-agent off-policy
correction, ensuring monotonic improvement for each agent update. A2PO also proposes a novel semi-greedy
strategy for sequential agent update ordering.

Example:

.. code-block:: shell

    python3 light_malib/main_pb.py --config expr_configs/cooperative_MARL_benchmark/academy/3_vs_1_with_keeper/a2po.yaml



Fully Decentralized Learning
-----------------------------------------------------------

Multi-Agent Transformer (MAT)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

MAT treats multi-agent reinforcement learning as a sequence modeling
problem, employing Transformer networks for both actors and critics. Its policy updating algorithm bears a
resemblance to IPPO and MAPPO. However, MAT is a fully-centralized algorithm that trains a single joint actor
with access to all local observations from each agent, predicting actions in an auto-regressive manner.

Example:

.. code-block:: shell

    python3 light_malib/main_pb.py --config expr_configs/cooperative_MARL_benchmark/academy/3_vs_1_with_keeper/mat.yaml