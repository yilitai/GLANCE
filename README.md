<p align="center">
  <img src="assets/logo.svg" alt="GLANCE logo" width="380">
</p>


<p align="center">
  <strong>Gaussian Latent Alignment via Neighbor Contrastive Encoding</strong>
</p>

<p align="center">
  Batch-aware integration and protein-level interpretation of single-cell mass spectrometry and spatial proteomics data.
</p>

## Overview

**GLANCE** is a probabilistic framework for integrating single-cell mass spectrometry and spatial proteomics data across batches while supporting interpretable protein-level analysis. Built on a conditional variational autoencoder, GLANCE learns a shared latent representation and derives **abundance scores (AS)** from its probabilistic decoder.

GLANCE incorporates a within-batch local manifold constraint to preserve biological structure and uses Procrustes alignment across independently trained models to obtain a consensus representation. A Student's *t* decoder supports robust modeling of protein abundance measurements and enables probabilistic protein-level interpretation.

## Key features

- **Batch-aware integration:** Learn a shared cell representation while accounting for batch-associated variation.
- **Preservation of biological structure:** Maintain local relationships and the distinctness of rare cell populations during integration.
- **Consensus representation:** Combine independently trained models using Procrustes alignment.
- **Protein-level interpretation:** Compute abundance scores as exceedance probabilities reflecting relative protein abundance levels within the overall population.
- **Downstream analysis:** Support cell clustering, protein correlation module discovery, and spatial niche identification in conjunction with downstream tools.

## Installation

We recommend using a dedicated Python 3.10 environment:

```bash
conda create -n glancevae python=3.10.19 -y
conda activate glancevae
```

**GPU support.** If you plan to use GPU acceleration, install the appropriate CUDA-enabled version of [PyTorch](https://pytorch.org/get-started/locally/) before installing GLANCE. For example, for CUDA 12.6:

```bash
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
```

**Install GLANCE.** Install GLANCE directly from GitHub:

```bash
pip install "glancevae @ git+https://github.com/yilitai/GLANCE.git"
```

To install the additional dependencies used in the tutorial notebooks:

```bash
pip install "glancevae[tutorials] @ git+https://github.com/yilitai/GLANCE.git"
```

These commands require Git to be installed.

## Tutorials

Step-by-step examples are available in the [`tutorials/`](tutorials/) directory. They cover a typical workflow, from preparing a cell-by-protein dataset and specifying batch labels to training GLANCE, examining the integrated representation, and interpreting abundance scores.

## Input and output

GLANCE works with cell-by-protein abundance data and corresponding batch labels in an [AnnData](https://anndata.readthedocs.io/) workflow.

| Input | Output |
| --- | --- |
| Cell-by-protein abundance matrix and batch annotations | Consensus latent representation for downstream cell-level analysis |
| | Abundance scores for protein-level interpretation |

See the tutorials for data preparation, model usage, and result visualization.

## Citation

A manuscript describing GLANCE will be linked here when a preprint or published version becomes available. If you use GLANCE in your research, please cite the corresponding paper once available.

## Support

For bug reports, questions, or feature requests, please [open a GitHub Issue](https://github.com/yilitai/GLANCE/issues).

## License

GLANCE is distributed under the [Apache License 2.0](LICENSE).
