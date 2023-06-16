Competitive Learning --- PBT
======================================================================

.. contents::
    :local:
    :depth: 2

----------------------

To achieve stronger and less exploitable policies, we embrace competitive learning as a general approach. Different from purely cooperative learning
which we defined previously as best responding to a fixed opponent under a pre-defined scenario, we want to acquire stronger skills in football by competing
with changing opponent for better generalization. Our framework is designed to seamlessly accommodate various self-play methods, including Naive
566 Self-Play (SP), Fictitious Self-Play (FSP), Policy Space Response Oracle (PSRO), and even
567 League Training [33]. By offering such flexibility, we enable researchers to explore and utilize the most suitable
568 self-play method for their specific needs.


Policy-Space Response Oracle (PSRO)
---------------------------------------------------------

To run a PSRO trial, execute the command:

.. code-block:: shell

    python3 light_malib/main_pb.py --config expr_configs/population_based_self_play/ippo_5v5_hard_psro.yaml




League Training
-----------------------------------------------------------
