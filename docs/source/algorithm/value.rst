Value-based Learning
======================================================================

.. contents::
    :local:
    :depth: 3

----------------------

.. _QMIX:

QMIX
---------------------------------------------------------
QMIX is a value-based cooperative algorithm that employs a mixing network to combine
individual Q functions into a joint Q function Qjoint in a complex non-linear way. The mixing network
structurally enforces the Individual-Global-Max (IGM) principle  that the joint-action value should be
monotonic in the per-agent values.

.. _QPLEX:

QPLEX
---------------------------------------------------------
QPLEX  also follows the IGM principle but applies its advantage-
based version. It has a Q-value mixing network similar to QMIX’s but incorporates dueling structures
for representing both individual and joint Q functions. In our experiments, we actually adopt their enhanced
versions proposed in the CDS paper, where an additional information-theoretical regularization is introduced
to promote exploration.


