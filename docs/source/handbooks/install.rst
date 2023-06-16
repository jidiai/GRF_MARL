.. _install:

Installation
============

You can use any tool to manage your python environment. Here, we use conda as an example.

.. code-block:: shell

    conda create -n marllib python==3.9  # or 3.8
    conda activate marllib

    git clone https://github.com/jidiai/GRF_MARL.git
    cd GRF_MARL
    pip install -r requirements.txt

* We strongly recommend install the Google Research Football dependencies first following the instruction presented in the `official website <https://github.com/google-research/football>`_

* We did not include Pytorch in our dependencies and you can install it following the `official website <https://pytorch.org/get-started/locally/>`_ as well. We recommand version ``<=1.13.0``
