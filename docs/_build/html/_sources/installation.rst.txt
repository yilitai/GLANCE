Installation
============

Python environment
------------------

The repository recommends a dedicated Python 3.10 environment:

.. code-block:: bash

   conda create -n glancevae python=3.10.19 -y
   conda activate glancevae

Install from GitHub
-------------------

Git must be installed for these commands.

.. code-block:: bash

   pip install "glancevae @ git+https://github.com/yilitai/GLANCE.git"

For the notebook examples, install the tutorial extras:

.. code-block:: bash

   pip install "glancevae[tutorials] @ git+https://github.com/yilitai/GLANCE.git"

For NVIDIA GPU training, install the appropriate CUDA-enabled version of
`PyTorch <https://pytorch.org/get-started/locally/>`_ before installing GLANCE.
CPU execution is also supported.
