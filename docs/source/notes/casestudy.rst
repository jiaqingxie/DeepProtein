Case Study  
================================================

There are many case studies where one single line is performed.


1a. Protein Function (Property) Prediction.**
^^^^^^^^^^^^^^^^^^^^^^^^^^^

We take Beta-lactamase dataset as an example.

CNN Case:

.. code-block:: bash

  $ python train/beta.py --target_encoding CNN --seed 7 --wandb_proj DeepProtein --lr 0.0001 --epochs 100

GNN Case:

.. code-block:: bash

  $ python train/beta.py --target_encoding DGL_GCN --seed 7 --wandb_proj DeepProtein --lr 0.00001 --epochs 100



1b. Protein Protein Interaction.**
^^^^^^^^^^^^^^^^^^^^^^^^^^^

We take PPI Affinity dataset as an example.

CNN Case:

.. code-block:: bash


  $ python train/ppi_affinity.py --target_encoding CNN --seed 42 --wandb_proj DeepProtein --lr 0.0001 --epochs 100

GNN Case:

.. code-block:: bash

  $ python train/ppi_affinity.py --target_encoding DGL_GCN --seed 42 --wandb_proj DeepProtein --lr 0.00001 --epochs 100





