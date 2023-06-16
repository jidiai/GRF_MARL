Centralized Training Decentralized Execution (CTDE)
======================================================================

.. contents::
    :local:
    :depth: 3

----------------------

.. _MAPPO:

Multi-Agent Proximal Policy Optimization (MAPPO)
---------------------------------------------------------
MAPPO follows the Centralized Training Decentralized Execution (CTDE) paradigm, enabling
actors to access local information while critics exploit global state information. Critics with more comprehensive
information typically yield better value estimation, thus resulting in improved policies.



.. _HAPPO:

Heterogeneous-Agent PPO (HAPPO)
---------------------------------------------------------
HAPPO and A2PO update agents sequentially to prevent the potential conflicting direction updates that may happen in the
simultaneous updating. HAPPO leverages the multi-agent advantage decomposition lemma and guarantees
a monotonic improvement on the joint policy.


.. _A2PO:

Agent-by-agent Policy Optimization (A2PO)
---------------------------------------------------------
A2PO derives an updating formula similar to PPO for each
agent and resolves the preceding agent policy shifting issue using a technique called preceding-agent off-policy
correction, ensuring monotonic improvement for each agent update. A2PO also proposes a novel semi-greedy
strategy for sequential agent update ordering.










