#! /bin/bash

# Please modify < INSTALL_DIR > to the absolute installation path of IIDL-PepPI.
IIDLPepPI = <INSTALL_DIR>

python test.py  -method_path $IIDLPepPI -pep_fasta $IIDLPepPI/example/example_peptide_1.fasta \
                -pro_fasta $IIDLPepPI/example/example_protein_1.fasta -csv_path $IIDLPepPI/example/
