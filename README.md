# IIDL-PepPI

<div align="center">
  
  [![GitHub stars](https://badgen.net/github/stars/ShutaoChen97/IIDL-PepPI)](https://GitHub.com/ShutaoChen97/IIDL-PepPI/stargazers/)
  [![GitHub watchers](https://badgen.net/github/watchers/ShutaoChen97/IIDL-PepPI/)](https://GitHub.com/ShutaoChen97/IIDL-PepPI/watchers/)
  [![GitHub forks](https://badgen.net/github/forks/ShutaoChen97/IIDL-PepPI/)](https://GitHub.com/ShutaoChen97/IIDL-PepPI/network/)
  [![GitHub issues](https://badgen.net/github/issues/ShutaoChen97/IIDL-PepPI/?color=red)](https://GitHub.com/ShutaoChen97/IIDL-PepPI/issues/)
  [![GitHub license](https://img.shields.io/github/license/ShutaoChen97/IIDL-PepPI.svg)](https://github.com/ShutaoChen97/IIDL-PepPI/blob/master/LICENSE)
  [![Zenodo DOI:10.5281/zenodo.10240298](https://zenodo.org/badge/DOI/10.1007/978-3-319-76207-4_15.svg)](https://doi.org/10.5281/zenodo.10240298)

</div>


Accurately predicting peptide-protein interactions (PepPIs) is essential for biology and disease research. Given the homogeneity of biological sequences and natural language, the grammar and semantics of peptides or proteins have been extensively studied to solve the important tasks in protein sequence analysis. However, these methods ignored the pragmatic information of proteins. Here, we introduce **IIDL-PepPI**, **a progressive transfer learning model based on pragmatic analysis** for predicting binary interactions and binding residues in peptide-protein-specific pairs.

**Given the complexity and instability of individuals in configuring the environment, we strongly recommend that users use IIDL-PepPI 's online prediction Web server, which can be accessed through **http://bliulab.net/IIDL-PepPI/**.**

![IIDL-PepPI](/imgs/IIDL-PepPI.png)
**Fig. 1: Data preparation workflow and network architecture of IIDL-PepPI. a** Data preparation workflow of IIDL-PepPI, in which the public databases used include RCSB PDB, PDBe, and UniProt. **b** Network architecture of IIDL-PepPI for peptide-protein binary interaction prediction and binding residue recognition, including sequence representation, feature encoding, bi-attentional module, and decoding. Based on the biological sequence pragmatic analysis, the bi-attention module explicitly integrates features from the peptide and protein sides to distinguish different peptide-protein-specific interactions. **c** The progressive transfer learning architecture. The initial stage of IIDL-PepPI commences with pre-training peptide-protein binary interactions using sequence-level datasets and the coarse-grained learning of basic network parameters. Subsequently, in the second phase, we transfer the parameters of the basic network, replace the decoder, and conduct fine-grained fine-tuning of the model using residue-level dataset for precise prediction of peptide- and protein-binding residues in specific peptide-protein pairs.

# 1 Installation

## 1.1 Create conda environment

```
conda create -n iidl python=3.10
conda activate iidl
```

## 1.2 Requirements
The main dependencies used in this project are as follows (for more information, please see the `environment.yaml` file):

```
python  3.10
biopython 1.81
huggingface-hub 0.19.4
numpy 1.26.2
pandas 2.1.3
scikit-learn 1.3.2
scipy 1.11.4
tokenizers 0.15.0
torch 2.1.1+cu118
torchaudio 2.1.1+cu118
tqdm 4.66.1
transformers 4.35.2
```

> **Note** If you have an available GPU, the accelerated IIDL-PepPI can be used to predict peptide-protein binary interactions and pair-specific binding residues. Change the URL below to reflect your version of the cuda toolkit (cu118 for cuda=11.6 and cuda 11.8, cu121 for cuda 12.1). However, do not provide a number greater than your installed cuda toolkit version!
> 
> ```
> pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118
> ```
>
> For more information on other cuda versions, see the [pytorch installation documentation](https://pytorch.org/).

## 1.3 Tools
Two multiple sequence alignment tools and three databases are required: 
```
SCRATCH-1D 1.2
IUPred2A \
ncbi-blast 2.13.0
ProtBERT \
```

Databases:
```
nrdb90 (http://bliulab.net/sAMPpred-GAT/static/download/nrdb90.tar.gz)
```

**nrdb90**: We have supplied the nrdb90 databases on our webserver. You need to put it into the `utils/` directoy and decompress it. 

> **Note** that all the defalut paths of the tools and databases are shown in `config.yaml`. You can change the paths of the tools and databases by configuring `config.yaml` as you need. 

`SCRATCH-1D`, `IUPred2A`, `ncbi-blast`, and `ProtBERT` are recommended to be configured as the system envirenment path. Your can follow these steps to install them:

### 1.3.1 How to install SCRATCH-1D
Download (For linux, about 6.3GB. More information, please see **https://download.igb.uci.edu/**)
```
wget https://download.igb.uci.edu/SCRATCH-1D_1.2.tar.gz
tar -xvzf SCRATCH-1D_1.2.tar.gz
```

Install
```
cd SCRATCH-1D_1.2
perl install.pl
```

> **Note:** The 32 bit linux version of blast is provided by default in the 'pkg' sub-folder of the package but can, probably should, and in some cases has to be replaced by the 64 bit or Mac OS version of the blast software for improved performances and compatibility on such systems.


Finally, test the installation of SCRATCH-1D
```
cd <INSTALL_DIR>/doc
../bin/run_SCRATCH-1D_predictors.sh test.fasta test.out 4
```

> **Note:** If your computer has less than 4 cores, replace 4 by 1 in the command line above.


### 1.3.2 How to install IUPred2A
For download and installation of IUPred2A, please refer to **https://iupred2a.elte.hu/download_new**. It should be noted that this automation service is **only applicable to academic users.** For business users, please contact the original authors for authorization.

After obtaining the IUPred2A software package, decompress it.
```
tar -xvzf iupred2a.tar.gz
```

Finally, test the installation of IUPred2A
```
cd <INSTALL_DIR>
python3 iupred2a P53_HUMAN.seq long
```


### 1.3.3 How to install ncbi-blast
Download (For x64-linux, about 220M. More information, please see **https://blast.ncbi.nlm.nih.gov/doc/blast-help/downloadblastdata.html**)
```
wget https://ftp.ncbi.nlm.nih.gov/blast/executables/blast+/2.13.0/ncbi-blast-2.13.0+-x64-linux.tar.gz
tar -xvzf ncbi-blast-2.13.0+-x64-linux.tar.gz
```
Add the path to system envirenment in `~/.bashrc`.
```
export BLAST_HOME={your_path}/ncbi-blast-2.13.0+
export PATH=$PATH:$BLAST_HOME/bin
```

Finally, reload the system envirenment and check the ncbi-blast command:
```
source ~/.bashrc
psiblast -h
```

> **Note:** The purpose of IIDL-PepPI with the help of ncbi-blast is to extract the position-specific scoring matrix (PSSM). It should be noted that for sequences that cannot be effectively aligned, the PSSM is further extracted by blosum62 (which can be found in `utils/blosum62.txt`).


### 1.3.4 How to install ProtBERT
Download and install (More information, please see **https://huggingface.co/Rostlab/prot_bert** or **https://github.com/agemagician/ProtTrans**)

```
wget https://zenodo.org/records/4633691/files/prot_bert.zip
```

or

```
git lfs install
git clone https://huggingface.co/Rostlab/prot_bert
```


## 1.4 Inatsll IIDL-PepPI
To install from the development branch run
```
git clone git@github.com:ShutaoChen97/IIDL-PepPI.git
cd IIDL-PepPI/utils
tar -xvzf nrdb90.tar.gz
cd ..
```

Besides, **due to the limit of 2G file size uploaded by Git LFS**, the comparison file used by IIDL-PepPI to reduce the dimensionality of pre-training (ProtBERT) features is available through our [IIDL-PepPI online Web server](http://bliulab.net/IIDL-PepPI) (about 3G).
```
wget http://bliulab.net/IIDL-PepPI/static/download/protein_webserver.pkl
mv protein_webserver.pkl saved_models/protbert_feature_before_pca/
```



**Finally, configure the Defalut path of the above tool and the database in `config.yaml`. You can change the path of the tool and database by configuring `config.yaml` as needed.**


# 2 Usage
It takes 2 steps to predict peptide-protein binary interaction and peptide-protein-specific binding residues:

(1) Replace the default peptide sequence in the `example/example_peptide_ 1.fasta` file with your peptide sequence (FASTA format). Similarly, replace the default protein sequence in the `example/example_protein_ 1.fasta` file with your protein sequence (FASTA format). If you don't want to do this, you can also test your own peptide-protein pairs by modifying the two sequence file paths passed in by the `test.sh` script (the two parameters are `-pep_fasta` for peptide and `-pro_fasta` for protein, respectively).

(2) Then, run `test.sh` to make multi-level prediction, including binary interaction prediction and combined residue recognition. 
It should be noted that `test.sh` automatically calls the scripts `generate_peptide_features.py`, `generate_protein_features.py`, and `generate_pssm.py` to generate the multi-source isomerization characteristics of peptides and proteins.
```
bash test.sh
```
 
> **Note** you can running `python test.py -h` to learn the meaning of each parameter.

If you want to retrain based on your private dataset, find the original IIDL-PepPI model in `model/IIDL-PepPI.py`. The IIDL-PepPI source code we wrote is based on the Pytorch implementation and can be easily imported by instantiating it.



# 3 Problem feedback
If you have questions on how to use IIDL-PepPI (or PDB-BRE), feel free to raise questions in the [discussions section](https://github.com/ShutaoChen97/IIDL-PepPI/discussions). If you identify any potential bugs, feel free to raise them in the [issuetracker](https://github.com/ShutaoChen97/IIDL-PepPI/issues).

In addition, if you have any further questions about IIDL-PepPI, please feel free to contact us [**stchen@bliulab.net** or **shutao.chen@bit.edu.cn**]

# 4 Citation

If you find our work useful, please cite us at
```
@article{chen2023pdb,
  title={PDB-BRE: A ligand--protein interaction binding residue extractor based on Protein Data Bank},
  author={Chen, Shutao and Yan, Ke and Liu, Bin},
  journal={Proteins: Structure, Function, and Bioinformatics},
  year={2023},
  publisher={Wiley Online Library}
}

@article{chen2024peptide,
  title={Peptide-Protein Interaction Profiling Based on Pragmatic Analysis and Progressive Transfer Learning},
  author={Chen, Shutao and Yan, Ke and Li, Xuelong and Liu, Bin},
  journal={submitted},
  year={2024},
  publisher={}
}

```
