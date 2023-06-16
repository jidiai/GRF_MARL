.. _architecture:

*******************************
Framework Architecture
*******************************


Here we introduce our framework architecture

.. contents::
    :local:
    :depth: 1


.. _workflow:

Workflow
=================================

.. figure:: ../images/framework.png
    :align: center
    :width: 700
    :alt: workflow


.. _rollout:

Rollout Manager
===================================
The Rollout Manager establishes multiple parallel rollout workers and
delegates rollout tasks to each worker. Each rollout task includes environment
settings, policy distributions for simulation, and information pertaining to the
Episode Server.


.. _train:

Training Manager
===================================
The Training Manager sets up multiple distributed trainers and assigns training
tasks to each trainer. Training task descriptions consist of training configurations
and details regarding the Policy and Episode buffers.


.. _buffer:

Data Buffer
==================================
The Data Buffer serves as a repository for episodes and policies. The Episode
Server saves new episodes submitted by the rollout workers, while trainers retrieve
sampled episodes from the Episode Server for training. The Policy Server, on the other
hand, stores updated policies submitted by the Training Manager. Rollout workers
subsequently fetch these updated policies from the Policy Server for simulation.


.. _agent_manager:

Agent Manager
==================================
The Agent Manager manages a population of policies and their associated data, which
includes pairwise match results and individual rankings.


.. _scheduler:

Task Scheduler
==================================
The Task Scheduler is responsible for scheduling and assigning tasks to the
Training Manager and Rollout Manager. In each training generation, it selects an opponent distribution
based on computed statistics retrieved from the Agent Manager.


.. _pbt:

Population-based Training Workflow
====================================
Beside training against a fixed opponent, Light-MALib also supports population-based training, such as Policy-Space Response Oracle (PSRO).
An illustration of a PSRO trial is given as below:

.. figure:: ../images/psro.svg
    :align: center
    :width: 500
    :alt: workflow







