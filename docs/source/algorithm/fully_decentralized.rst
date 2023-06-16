Fully Decentralized Learning
======================================================================

.. contents::
    :local:
    :depth: 3

----------------------

.. _MAT:

Multi-Agent Transformer (MAT)
---------------------------------------------------------
MAT treats multi-agent reinforcement learning as a sequence modeling
problem, employing Transformer networks for both actors and critics. Its policy updating algorithm bears a
resemblance to IPPO and MAPPO. However, MAT is a fully-centralized algorithm that trains a single joint actor
with access to all local observations from each agent, predicting actions in an auto-regressive manner.


