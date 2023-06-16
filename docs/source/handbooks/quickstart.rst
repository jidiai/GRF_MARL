.. _quick-start:

Quick Start Benchmarking
===========

.. contents::
    :local:
    :depth: 2

Execution
-----------------
To run an experiment, simple execute the following command with configuration file path input

.. code-block:: shell

    python3 light_malib/main_pb.py --config expr_configs/xxx/.../xxx.yaml

All the configuration files can be found in ``.expr_configs/``, including those for academy scenarios, full-game scenarios and a PSRO trial.



Configuration file
-----------------
Each configuration file define detailed settings for rollout, training, data storage, logging, model setup, population and more.

Framework Config
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The framework section defines type of learning

.. list-table::
   :widths: 25 25
   :align: center
   :header-rows: 0

   * - ``max_round``
     - maximum number of generation (only in PBT)
   * - ``meta_solver``
     - type of meta solver (only in PBT)
   * - ``sync_training``
     - synchronous if True else asynchronous mode
   * - ``stopper.max_steps``
     - maximum number of rollout iteration


Rollout Manager Config
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The rollout manager section defines rollout settings

.. list-table::
   :widths: 25 25
   :align: center
   :header-rows: 0

   * - ``num_worker``
     - number of parallel rollout worker
   * - ``batch_size``
     - size of data batch collected
   * - ``rollout_length``
     - rollout episode maximum length
   * - ``sample_length``
     - truncation length (no trucation if 0)
   * - ``envs``
     - configs of the environment



Training Manager Config
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The training manager section defines training settings

.. list-table::
   :widths: 25 25
   :align: center
   :header-rows: 0

   * - ``batch_size``
     - training batch size
   * - ``num_trainers``
     - number of trainers (or GPU number)
   * - ``rollout_length``
     - rollout episode maximum length
   * - ``optimizer``
     - optimizer type
   * - ``actor_lr``
     - learning rate of the actor
   * - ``critic_lr``
     - learning rate of the critic


DataServer Config
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The data server section defines data storage

.. list-table::
   :widths: 25 25
   :align: center
   :header-rows: 0

   * - ``capacity``
     - data table size
   * - ``sampler_type``
     - data sampling scheme
   * - ``sample_max_usage``
     - maximum reusage of each sample


Population Config
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The population section define the whole population, including the trainable policies. For each algorithm:

**Model config**:

.. list-table::
   :widths: 25 25
   :align: center
   :header-rows: 0

   * - ``model``
     - model type (actor critic type, feature encoder type)
   * - ``initialization``
     - model initialization
   * - ``actor``
     - actor network setting
   * - ``critic``
     - critic network setting

**Custom config**:

.. list-table::
   :widths: 25 25
   :align: center
   :header-rows: 0

   * - ``FE_cfg``
     - configs for feature encoder
   * - ``gamma``
     - discount value
   * - ``ppo_epoch``
     - training epoch

**Policy init cfg**

.. list-table::
   :widths: 25 25
   :align: center
   :header-rows: 0

   * - ``agent_0``
     - population pool for agent 0
   * - ``init_cfg``
     - how agent 0 is initialize in each condition (random, pretrained or interit)
   * - ``agent_1``
     - population pool for agent 1



Monitoring & Saving
-----------------
The learning statistics will be recorded in ``./logs`` path, including the tensorboard file, saved config file, saved policies and others.







