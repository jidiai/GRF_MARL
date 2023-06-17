Pre-trained Policies
======================================================================

.. contents::
    :local:
    :depth: 3

----------------------

We release some research tools that we find very useful during our experiment with GRF. We wish the community can benefit from them
and continue their good works on MARL.

Here we provide our pre-trained models on both 5-vs-5 and 11-vs-11 full-game multi-agent scenarios. A good pre-trained models can be used for
various purpose:
#. As a generally strong opponent to compete with;
#. As a good model initialization to continue training on, saving skills learning time;
#. As a quality data generator which yield data with good exploration;

These models can be found in ``light_malib/trained_models/gr_football/``. Each corresponding folder contains three files: the actor, the critic and the model description. You can
write the path of the file into the ``population.init_policies`` section to use it as the initialization or opponents. Or you can perform **behaviour cloning** using our `light_malib/algorithm/bc` algorithm.

To give a comprehensive understanding of these pre-trained policies. We gather them together and do massive amount of simulation. We collect each piece of football statistics and rank the pre-trained policies using
radar plot:

5-vs-5
^^^^^^^^^^^^^^^^^^^^^^

.. figure:: ../images/radar_5v5.svg
    :align: center
    :width: 400
    :alt: psro img

    5-vs-5 pre-trained policies


11-vs-11
^^^^^^^^^^^^^^^^^^^^^^

.. figure:: ../images/radar_11v11.svg
    :align: center
    :width: 400
    :alt: psro img

    11-vs-11 pre-trained policies
