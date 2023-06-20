.. _install:

Installation
============

You can use any tool to manage your python environment. Here, we use conda as an example.

.. code-block:: shell

    conda create -n marllib python==3.9  # or 3.8
    conda activate marllib

Before installing the codebase, it is recommanded to install the Google Research Football dependencies first following the
instruction presented in the `official website <https://github.com/google-research/football>`_. On Linux, you can do it by:

.. code-block:: shell

    sudo apt-get update
    # install package
    sudo apt-get install git cmake build-essential libgl1-mesa-dev libsdl2-dev \
    libsdl2-image-dev libsdl2-ttf-dev libsdl2-gfx-dev libboost-all-dev \
    libdirectfb-dev libst-dev mesa-utils xvfb x11vnc python3-pip

    python3 -m pip install --upgrade pip setuptools psutil wheel
    # install env
    python3 -m pip install gfootball

Run the following command to make sure you have installed the GRF correctly:

.. code-block:: shell

    python3 -m gfootball.play_game

Next, install the GRF-MARLLib:

.. code-block:: shell

    git clone https://github.com/jidiai/GRF_MARL.git
    cd GRF_MARL
    pip install -r requirements.txt


* We did not include Pytorch in our dependencies and you can install it following the `official website <https://pytorch.org/get-started/locally/>`_ as well. We recommand version ``<=1.13.0``
