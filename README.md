# IIDL-PepPI

Accurately predicting peptide-protein interactions (PepPIs) is essential for biology and disease research. Given the homogeneity of biological sequences and natural language, the grammar and semantics of peptides or proteins have been extensively studied to solve the important tasks in protein sequence analysis. However, these methods ignored the pragmatic information of proteins. Here, we introduce **IIDL-PepPI**, **an interpretable progressive transfer learning model based on biological sequence pragmatic analysis** for predicting binary interactions and binding residues in peptide-protein-specific pairs.

The online prediction webserver of IIDL-PepPI can be accessible at **http://bliulab.net/IIDL-PepPI/**.

![IIDL-PepPI](/imgs/IIDL-PepPI.png)
**Fig. 1: Data preparation workflow and network architecture of IIDL-PepPI. a** Data preparation workflow of IIDL-PepPI, in which the public databases used include RCSB PDB, PDBe, and UniProt. **b** Network architecture of IIDL-PepPI for peptide-protein binary interaction prediction and binding residue recognition, including sequence representation, feature encoding, bi-attentional module, and decoding. Based on the biological sequence pragmatic analysis, the bi-attention module explicitly integrates features from the peptide and protein sides to distinguish different peptide-protein-specific interactions. **c** The progressive transfer learning architecture. The initial stage of IIDL-PepPI commences with pre-training peptide-protein binary interactions using sequence-level datasets and the coarse-grained learning of basic network parameters. Subsequently, in the second phase, we transfer the parameters of the basic network, replace the decoder, and conduct fine-grained fine-tuning of the model using residue-level dataset for precise prediction of peptide- and protein-binding residues in specific peptide-protein pairs.

# Installation

IIDL-PepPI can be installed from pip via

```
pip install deepblast
```

To install from the development branch run

```
pip install git+https://github.com/flatironinstitute/deepblast.git
```

# Downloading pretrained models and data

The pretrained DeepBLAST model can be downloaded [here](https://users.flatironinstitute.org/jmorton/public_www/deepblast-public-data/checkpoints/deepblast-l8.ckpt).

The TM-align structural alignments used to pretrain DeepBLAST can be found below
- [Training data](https://users.flatironinstitute.org/jmorton/public_www/deepblast-public-data/train_matched.txt)
- [Validation data](https://users.flatironinstitute.org/jmorton/public_www/deepblast-public-data/valid.txt)
- [Testing data](https://users.flatironinstitute.org/jmorton/public_www/deepblast-public-data/test.txt)


See the [Malisam](http://prodata.swmed.edu/malisam/) and [Malidup](http://prodata.swmed.edu/malidup/) websites to download their datasets.

# Getting started

See the [wiki](https://github.com/flatironinstitute/deepblast/wiki) on how to use DeepBLAST and TM-vec for remote homology search and alignment.
If you have questions on how to use DeepBLAST and TM-vec, feel free to raise questions in the [discussions section](https://github.com/ShutaoChen97/IIDL-PepPI/discussions). If you identify any potential bugs, feel free to raise them in the [issuetracker](https://github.com/ShutaoChen97/IIDL-PepPI/issues).

# Citation

If you find our work useful, please cite us at
```
@article{chen2023pdb,
  title={PDB-BRE: A ligand--protein interaction binding residue extractor based on Protein Data Bank},
  author={Chen, Shutao and Yan, Ke and Liu, Bin},
  journal={Proteins: Structure, Function, and Bioinformatics},
  year={2023},
  publisher={Wiley Online Library}
}

@article{chen2023interpretable,
  title={Interpretable Progressive Transfer Learning for Peptide-Protein-Specific Interaction Profiling based on Biological Sequence Pragmatic Analysis},
  author={Chen, Shutao and Yan, Ke and Liu, Bin},
  journal={submitted},
  year={2023},
  publisher={}
}

```
