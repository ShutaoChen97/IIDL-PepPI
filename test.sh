#! /bin/bash

# IIDLPepPI=/home/cst/GitHub/IIDL-PepPI
IIDLPepPI = <INSTALL_DIR>

python test.py  -method_path $IIDLPepPI -pep_fasta $IIDLPepPI/example/example_peptide_1.fasta \
                -pro_fasta $IIDLPepPI/example/example_protein_1.fasta -csv_path $IIDLPepPI/example/
