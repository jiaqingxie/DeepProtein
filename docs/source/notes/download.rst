Download Code & Install
========================================================================


Download Code
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


.. code-block:: bash

   $ git clone https://github.com/jiaqingxie/DeepProtein.git
   $ ###  Download code repository 
   $
   $
   $ cd DeepProtein
   $ ### Change directory to DeepProtein



First time usage: setup conda environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

   $ ## Build virtual environment with all packages installed using conda
   $ conda create -n DeepProtein python=3.9
   $  
   $ ##  Activate conda environment 
   $ conda activate DeepProtein
   $ 
   $ ## Install necessary packages
   $ pip install git+https://github.com/bp-kelley/descriptastorus
   $ pip install lmdb seaborn wandb pydantic DeepPurpose
   $ conda install -c conda-forge pytdc
   $ 
   $
   $ ## Choice 1: Torch 2.3.0 + CUDA Version 11.8
   $ pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
   $ pip install torch-geometric
   $ 
   $ ## Choice 2: Torch 2.3.0 + CPU
   $ pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cpu
   $ pip install torch-geometric
   $ 
   $ pip install -r requirements.txt
   $ conda deactivate ### exit

DeepProtein 2.0 phase 1 removes DGL from the core installation path. The legacy graph encoders are being migrated to a torch-geometric backend, so the currently supported v2 workflows are the torch-only sequence and language-model encoders.


Another Choice is to use python virtual env where it have saved plenty of space on package management.

.. code-block:: bash


   $ ## cd to the expected path where you want to create such python env.
   $ cd env_path
   $
   $ ## Build virtual python env.
   $ python -m venv DeepProtein
   $  
   $ ## You will find a folder named DeepProtein under your current env_path
   $ ## Then you source the activate similar to conda activate:
   $ source env_path/DeepProtein/activate
   $ 
   $ ## Your will see sth. like (DeepProtein)(base) and this means the env is correctly activated
   $ ## Install the packages as above
   $
   $ ## cd DeepProtein and pip install the rest packages
   $ cd DeepProtein
   $ pip install -r requirements.txt
   $ deactivate ### exit


Second time and later
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you use conda env, then 

.. code-block:: bash

   $ conda activate DeepProtein
   $ ##  Activate conda environment
   $
   $
   $ conda deactivate ### exit

If you use python virtual env, then 

.. code-block:: bash

   $ source env_path/DeepProtein/activate
   $ ##  Activate python virtual environment where you saved it.
   $ ##  In default we assume you use Linux / MacOS, otherwise remove "source"
   $ ##  Just:
   $ env_path/DeepProtein/activate
   $
   $ conda deactivate ### exit
