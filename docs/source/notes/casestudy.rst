Case Study  
================================================

There are many case studies where one single line is performed. Suppose you are under the main folder of DeepProtein where it
contains a folder called train.


1a. Protein Function (Property) Prediction
^^^^^^^^^^^^^^^^^^^^^^^^^^^

We take Beta-lactamase dataset as an example.

CNN Case:

.. code-block:: bash

  $ python train/beta.py --target_encoding CNN --seed 7 --wandb_proj DeepProtein --lr 0.0001 --epochs 100

GNN Case:

.. code-block:: bash

  $ python train/beta.py --target_encoding DGL_GCN --seed 7 --wandb_proj DeepProtein --lr 0.00001 --epochs 100



1b. Protein Protein Interaction
^^^^^^^^^^^^^^^^^^^^^^^^^^^

We take PPI Affinity dataset as an example.

CNN Case:

.. code-block:: bash


  $ python train/ppi_affinity.py --target_encoding CNN --seed 42 --wandb_proj DeepProtein --lr 0.0001 --epochs 100

GNN Case:

.. code-block:: bash

  $ python train/ppi_affinity.py --target_encoding DGL_GCN --seed 42 --wandb_proj DeepProtein --lr 0.00001 --epochs 100


1c. Protein Localization Prediction
^^^^^^^^^^^^^^^^^^^^^^^^^^^

We take SubCellular dataset as an example.

CNN Case:

.. code-block:: bash


  $ python train/subcellular.py --target_encoding CNN --seed 7 --wandb_proj DeepProtein --lr 0.0001 --epochs 100

GNN Case:

.. code-block:: bash

  $ python train/subcellular.py --target_encoding DGL_GCN --seed 7 --wandb_proj DeepProtein --lr 0.00001 --epochs 100



1d.  Antigen Epitope Prediction
^^^^^^^^^^^^^^^^^^^^^^^^^^^
We take IEDB dataset as an example.

CNN Case:

.. code-block:: bash


  $ python train/IEDB.py --target_encoding Token_CNN --seed 7 --wandb_proj DeepProtein --lr 0.0001 --epochs 100



1e.  Antibody Paratope Prediction
^^^^^^^^^^^^^^^^^^^^^^^^^^^
We take SAbDab Liberis dataset as an example.

CNN Case:

.. code-block:: bash


  $ python train/SAbDab_Liberis.py --target_encoding Token_CNN --seed 7 --wandb_proj DeepProtein --lr 0.0001 --epochs 100


1f. Antibody Developability Prediction (TAP)
^^^^^^^^^^^^^^^^^^^^^^^^^^^
We take TAP dataset as an example.

CNN Case:

.. code-block:: bash


  $ python train/TAP.py --target_encoding CNN --seed 7 --wandb_proj DeepProtein --lr 0.0001 --epochs 100




