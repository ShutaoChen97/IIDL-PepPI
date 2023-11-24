# -*- coding: utf-8 -*-
import os
import csv
import torch
import argparse
from model.IIDLPepPI import IIDLPepPI, IIDLPepPIRes
from generate_peptide_features import PepFeature
from generate_protein_features import ProFeature
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
 
# %%
def BinaryPrediction(x_pep, x_ss_pep, x_2_pep, x_diso_pep, x_bert_pep, 
                     x_p, x_ss_p, x_2_p, x_dense_p, x_bert_p, models_save):
    # load binary prediction model
    model = IIDLPepPI().to(device)
    model_name = os.path.join(models_save, "IIDLPepPI_BinaryInteraction.pth")
    model.load_state_dict(torch.load(model_name, map_location='cpu'))
    # eval mode
    model.eval()
    ouputs = model(x_pep, x_ss_pep, x_2_pep, x_diso_pep, x_bert_pep, 
                   x_p, x_ss_p, x_2_p, x_dense_p, x_bert_p)
    # Binary prediction result
    ouputs = ouputs[0][0].item()
    return ouputs

# %%
def ResiduePrediction(x_pep, x_ss_pep, x_2_pep, x_diso_pep, x_bert_pep, 
                      x_p, x_ss_p, x_2_p, x_dense_p, x_bert_p, models_save):
    # load binding residues prediction model
    model = IIDLPepPIRes().to(device)
    model_name = os.path.join(models_save, "IIDLPepPI_BindingResidue.pth")
    model.load_state_dict(torch.load(model_name, map_location='cpu'))
    # eval mode
    model.eval()
    ouputs = model(x_pep, x_ss_pep, x_2_pep, x_diso_pep, x_bert_pep, 
                   x_p, x_ss_p, x_2_p, x_dense_p, x_bert_p)
    ouputs = ouputs.cpu().detach().numpy()
    # Binding residues prediction result
    peptide_residue = ouputs[0:, 0:50]
    protein_residue = ouputs[0:, 50:800]
    return peptide_residue, protein_residue

# %%
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # method
    parser.add_argument('-method_path', type=str, default='IIDL-PepPI/',
                        help='The file path of the IIDL-PepPI model installation')
    # data and result
    parser.add_argument('-pep_fasta', type=str, default='example/example_peptide_1.fasta',
                        help='Fasta file path of peptide sequence')
    parser.add_argument('-pro_fasta', type=str, default='example/example_protein_1.fasta',
                        help='Fasta file path of protein sequence')
    parser.add_argument('-csv_path', type=str, default='example/',
                        help='File path of IIDL-PepPI analysis output')
    # tools
    parser.add_argument('-scratch_path', type=str, default='utils/SCRATCH-1D_1.2/',
                        help='The file path of SCRATCH-1D_1.2')
    parser.add_argument('-protbert_path', type=str, default='utils/prot_bert/',
                        help='The file path of prot_bert')
    parser.add_argument('-iupred2a_path', type=str, default='utils/iupred2a/',
                        help='The file path of iupred2a')
    parser.add_argument('-ncbiblast_path', type=str, default='utils/ncbi-blast-2.13.0+/',
                        help='The file path of ncbi-blast')
    parser.add_argument('-nrdb90_path', type=str, default='utils/nrdb90/',
                        help='The file path of NRDB90 databases')
    args = parser.parse_args()
    
    method_path = args.method_path
    pep_uip = args.pep_fasta
    pro_uip = args.pro_fasta
    csv_path = args.csv_path
    scratch = args.scratch_path
    protbert = args.protbert_path
    iupred2a = args.iupred2a_path
    ncbiblast = args.ncbiblast_path
    nrdb90 = args.nrdb90_path
    models_save = os.path.join(method_path, "saved_models")
   
    uip_tmp = pep_uip.rsplit('/', 1)
    if len(uip_tmp) == 2:
        uip = uip_tmp[0]

    pep_seq_list, pro_seq_list = [], []
    name_list = []

    for i in open(pep_uip):
        if i[0] != '>':
            pep_seq_list.append(i.strip())
    for i in open(pro_uip):
        if i[0] != '>':
            pro_seq_list.append(i.strip())

    # Extracting the representation of peptide
    x_pep, x_ss_pep, x_2_pep, x_diso_pep, x_bert_pep = PepFeature(pep_seq_list, pep_uip, 
                                                                  method_path, scratch, 
                                                                  protbert, iupred2a,
                                                                  device)
    # Extracting the representation of protein
    x_p, x_ss_p, x_2_p, x_dense_p, x_bert_p = ProFeature(pro_seq_list, pro_uip,
                                                         method_path, scratch, 
                                                         protbert, iupred2a,
                                                         ncbiblast, nrdb90,
                                                         device)
    
    # Peptide-protein binary interaction prediction
    ouputs = BinaryPrediction(x_pep, x_ss_pep, x_2_pep, x_diso_pep, x_bert_pep, 
                              x_p, x_ss_p, x_2_p, x_dense_p, x_bert_p, models_save)
    
    # Continue to predict binding residues if there is an peptide-protein binary interaction
    if round(ouputs) == 1:
        # Binding residues prediction of peptide and protein
        peptide_residue, protein_residue = ResiduePrediction(x_pep, x_ss_pep, x_2_pep, x_diso_pep, x_bert_pep, 
                                                             x_p, x_ss_p, x_2_p, x_dense_p, x_bert_p, models_save)
        peptide_residue = list(peptide_residue[0])
        protein_residue = list(protein_residue[0])
        for tmp in range(len(peptide_residue)):
            peptide_residue[tmp] = round(peptide_residue[tmp])
        for tmp in range(len(protein_residue)):
            protein_residue[tmp] = round(protein_residue[tmp])
    
    # Export the results to a csv file
    with open(os.path.join(csv_path, "result.csv"), 'a', newline='') as file:  
        writer = csv.writer(file)
        writer.writerow(["PeptideSeq", "ProteinSeq", 
                         "BinaryScore", "BinaryInteraction",
                         "PeptideBinding", "ProteinBinding"])
        
        pep_len = len(pep_seq_list[0])
        pro_len = len(pro_seq_list[0])
        if pro_len > 800:
            pro_len = 800
        
        peptide_lab_str = ''
        for tmp in range(pep_len):
            peptide_lab_str += str(peptide_residue[tmp])
        protein_lab_str = ''
        for tmp in range(pro_len):
            protein_lab_str += str(protein_residue[tmp])
            
        if round(ouputs) == 1:
            log_idx = True
            writer.writerow([str(pep_seq_list[0]), str(pro_seq_list[0]),
                             str(f"{ouputs:.3f}"), log_idx, 
                             str(peptide_lab_str), str(protein_lab_str)])
        else:
            log_idx = False
            writer.writerow([str(pep_seq_list[0]), str(pro_seq_list[0]),
                             str(f"{ouputs:.3f}"), log_idx, 
                             'NA', 'NA'])
    
    print('The analysis was successfully completed! The result has been saved to the result.csv.')
