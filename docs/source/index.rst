.. Light-MALib documentation master file, created by
   sphinx-quickstart on Fri Jun 16 02:49:01 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

A Google Research Football Multi-Agent Reinforcement Learning Library - Light-MALib
=======================================
This repo provides a simple, distributed and asynchronous multi-agent reinforcement learning framework for
the `Google Research Football <https://github.com/google-research/football>`_ environment, along with research tools and results for benchmarking.
In particular, it includes:

#. A distributed and asynchronous MARL framework
#. Implementation of algorithm IPPO, MAPPO, HAPPO, A2PO, MAT
#. Ready-to-run experiment configuration
#. Population-based training pipline, such as PSRO and League Training
#. Pre-trained GRF policies in both 5-vs-5 and 11-vs-11 full-game scenarios
#. Single-step match replay debugger
#. Tutorial for GRF online ranking


.. toctree::
   :maxdepth: 1
   :caption: Light-MALib Handbook

   handbooks/install
   handbooks/scenarios
   handbooks/architecture
   handbooks/quickstart


.. toctree::
   :maxdepth: 2
   :caption: Algorithm Documentation

   algorithm/algo


