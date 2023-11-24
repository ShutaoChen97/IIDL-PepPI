# -*- coding: utf-8 -*-
import os
import time
import torch
import subprocess
import numpy as np
from utils.PreTrain import PreTrain

def Batch_Feature(seqlist, featuredict, featurelen, mode):
    featurelist = []
    for tmp in range(len(seqlist)):
        feature = featuredict[seqlist[tmp]]
        if len(feature) < featurelen:
            diff = featurelen - len(feature)
            diffarr = np.zeros((diff, feature.shape[1]))
            featureuse = np.vstack((feature, diffarr))
        else:
            featureuse = feature[0:featurelen, ]
        featurelist.append(featureuse)
    if mode == 'float':
        featuretensor = torch.as_tensor(np.array(featurelist)).float()
    else:
        featuretensor = torch.as_tensor(np.array(featurelist)).long()
    return featuretensor

# Sequence integer encoding
def peptide_feature_dict(peplist):
    AminoAcidDic = dict(A=1, C=2, E=4, D=5, G=6, F=7, I=8, H=9, K=10, M=11, L=12, 
                        N=14, Q=15, P=16, S=17, R=18, T=20, W=21, V=22, Y=23, X=24)
    peptide_feature_dict = {}
    for tmp1 in range(len(peplist)):
        seqlist = []
        for tmp2 in range(50):
            if tmp2 < len(peplist[tmp1]):
                try:
                    value = AminoAcidDic[peplist[tmp1][tmp2]]
                except:
                    value = 24
            else:
                value = 0
            seqlist.append(value)
        peptide_feature_dict[peplist[tmp1]] = np.array(seqlist)
    out = Batch_Feature(peplist, peptide_feature_dict, 50, 'long')
    return out

# Physical and chemical properties
def peptide_2_feature_dict(peplist):
    AminoAcidDic = dict(A=1, R=6, N=4, D=5, C=3, Q=4, E=5, G=2, H=6, I=1, L=1, 
                        K=6, M=1, F=1, P=1, S=4, T=4, W=2, Y=4, V=1, X=7)
    peptide_2_feature_dict = {}
    for tmp1 in range(len(peplist)):
        seqlist = []
        for tmp2 in range(50):
            if tmp2 < len(peplist[tmp1]):
                try:
                    value = AminoAcidDic[peplist[tmp1][tmp2]]
                except:
                    value = 7
            else:
                value = 0
            seqlist.append(value)
        peptide_2_feature_dict[peplist[tmp1]] = np.array(seqlist)  
    out = Batch_Feature(peplist, peptide_2_feature_dict, 50, 'long')
    return out

# Disorder
def cmd(command):
    print(command)
    os.system(command)
    
def IUPred2A(pep_uip, method_path, iupred2a):
    pep_dir = pep_uip
    
    user_dir_tmp = pep_uip.rsplit('/', 1)
    if len(user_dir_tmp) == 2:
        user_dir = user_dir_tmp[0]
    
    user_dir = os.path.join(user_dir, "tmp")
    os.makedirs(user_dir, exist_ok = True)
    
    out_long_dir = os.path.join(user_dir, "Peptide_IUPred2A_long.txt")
    out_short_dir = os.path.join(user_dir, "Peptide_IUPred2A_short.txt")
    out_glob_dir = os.path.join(user_dir, "Peptide_IUPred2A_glob.txt")
    
    files = [out_long_dir, out_short_dir, out_glob_dir]
    for file_tmp in files:
        if os.path.exists(file_tmp):
            try:
                os.remove(file_tmp)
            except:
                None
 
    command_1 = os.path.join(method_path, "utils/iupred2a.sh ") + \
        os.path.join(iupred2a, "iupred2a.py") + ' ' + pep_dir + ' long ' + out_long_dir
    
    command_2 = os.path.join(method_path, "utils/iupred2a.sh ") + \
        os.path.join(iupred2a, "iupred2a.py") + ' ' + pep_dir + ' short ' + out_short_dir
    
    command_3 = os.path.join(method_path, "utils/iupred2a.sh ") + \
        os.path.join(iupred2a, "iupred2a.py") + ' ' + pep_dir + ' glob ' + out_glob_dir
    
    command = command_1 + ' & ' + command_2 + ' & ' + command_3
    process = subprocess.Popen(command, shell=True)
    process.wait()
    
    permissions = 0o777
    os.chmod(out_long_dir, permissions)
    os.chmod(out_short_dir, permissions)
    os.chmod(out_glob_dir, permissions)
    
def read_sspro_result(pssm_file, types):
    with open(pssm_file, 'r') as f:
        lines = f.readlines()
        if types == 'long' or types == 'short':
            lines = lines[7:]
        else:
            tmpidx = 0
            for line in lines:
                if line.strip('\n') == '# POS\tRES\tIUPRED2':
                    tmpidx += 1
                    break
                else:
                    tmpidx += 1
                    continue
            lines = lines[tmpidx:]
        seq = []
        mat = []
        for line in lines:
            tmp = line.strip('\n').split()
            seq.append(tmp[1])
            tmp = tmp[2]
            mat.append(tmp)
        mat = np.array(mat)
        mat = mat.astype(float)
    return seq, mat

def peptide_dense_feature_dict(peplist, pep_uip, method_path, iupred2a):
    IUPred2A(pep_uip, method_path, iupred2a)
    time.sleep(30)
    
    uip_tmp = pep_uip.rsplit('/', 1)
    if len(uip_tmp) == 2:
        uip = uip_tmp[0]
    uip = os.path.join(uip, "tmp")
    
    peptide_dense_feature_IUPred2A_dict = {}
    types = ['long', 'short', 'glob']
    for tmp1 in range(len(peplist)):
        listuse = []
        for typetmp in types:
            filepath = os.path.join(uip, str('Peptide_IUPred2A_' + typetmp + '.txt'))
            listuse.append(read_sspro_result(filepath, typetmp)[1])
        peptide_dense_feature_IUPred2A_dict[peplist[tmp1]] = np.array(listuse)
    
    dense_feature_dict = {}
    for tmp in range(len(peptide_dense_feature_IUPred2A_dict)):
        seq = list(peptide_dense_feature_IUPred2A_dict)[tmp]
        peptide_dense_feature = peptide_dense_feature_IUPred2A_dict[seq].T
        if len(peptide_dense_feature) < 50:
            diff = 50 - len(peptide_dense_feature)
            diffarr = np.zeros((diff, peptide_dense_feature.shape[1]))
            use = np.vstack((peptide_dense_feature, diffarr))
        else:
            use = peptide_dense_feature[0:50, ]
        if use.shape[0] == 50:
            dense_feature_dict[seq] = np.array(use)
        else:
            print('error!')
            break
    
    out = Batch_Feature(peplist, dense_feature_dict, 50, 'float')
    return out

# Secondary Structure
def Read_SeqID(FilePath):
    f0 = open(FilePath, 'r')
    lines = f0.readlines()
    count = 0
    info1 = []
    info2 = []
    for line in lines:
        if count % 2 == 0:
            info1.append(line.strip('\n').strip('>'))
        else:
            info2.append(line.strip('\n'))
        count += 1
    f0.close()
    return info1, info2

def peptide_ss_feature_dict(peplist, pep_uip, method_path, scratch):
    pep_dir = pep_uip
    
    uip_tmp = pep_uip.rsplit('/', 1)
    if len(uip_tmp) == 2:
        uip = uip_tmp[0]
        
    user_dir = os.path.join(uip, "tmp")
    os.makedirs(user_dir, exist_ok = True)
    out_dir = os.path.join(user_dir, "Peptide.out")
    scratch_bin = os.path.join(scratch, "bin/run_SCRATCH-1D_predictors.sh")
    command = os.path.join(method_path, "utils/SCRATCH.sh ") + scratch_bin + ' ' + pep_dir + ' ' + out_dir
    process = subprocess.Popen(command, shell=True)
    process.wait()
    
    AminoAcidDic = dict(A=1, C=2, E=4, D=5, G=6, F=7, I=8, H=9, K=10, M=11, L=12, N=14, 
                        Q=15, P=16, S=17, R=18, T=20, W=21, V=22, Y=23, X=24)

    filepath = 'Peptide.out.ss'
    _, sspro = Read_SeqID(os.path.join(user_dir, filepath))
    _, seq = Read_SeqID(pep_dir)

    amino = list(AminoAcidDic)
    AminoAcidDic_SSPro = {}
    SSProEle = ['H', 'C', 'E']
    for tmp1 in range(len(AminoAcidDic)):
        aminouse = amino[tmp1]
        embeddinguse = AminoAcidDic[aminouse]
        for tmp2 in range(len(SSProEle)):
            SSProEleUse = SSProEle[tmp2]
            keyuse = aminouse + SSProEleUse
            if keyuse not in AminoAcidDic_SSPro:
                AminoAcidDic_SSPro[keyuse] = embeddinguse * 3 - (2 - tmp2)
    
    peplenuse = 50
    ss_feature_dict = {}
    for tmp1 in range(len(seq)):
        sequse = seq[tmp1]
        ssprouse = sspro[tmp1]
        seqlist = []
        seqtmp = []
        for tmp2 in range(len(sequse)):
            seqtmp.append(sequse[tmp2] + ssprouse[tmp2])
            seqlist.append(AminoAcidDic_SSPro[sequse[tmp2]+ssprouse[tmp2]])
        seqlist = np.array(seqlist)
        if len(seqlist) < peplenuse:
            diff = peplenuse - len(seqlist)
            suparr = np.zeros((diff, ), dtype=float)
            seqlist = np.hstack((seqlist, suparr))
        elif len(seqlist) > peplenuse:
            seqlist = seqlist[0:peplenuse]
        seqtmp = ','.join(seqtmp)
        ss_feature_dict[seq[tmp1]] = np.array(seqlist)
       
    out = Batch_Feature(peplist, ss_feature_dict, 50, 'long')
    return out

# Peptide representation extraction scripts
def PepFeature(peplist, pep_uip, method_path, scratch, protbert, iupred2a, device): 
    x_pep = peptide_feature_dict(peplist)
    x_ss_pep = peptide_ss_feature_dict(peplist, pep_uip, method_path, scratch)
    x_2_pep = peptide_2_feature_dict(peplist)
    x_bert_pep = PreTrain(peplist, pep_uip, 50, method_path, protbert)
    x_diso_pep = peptide_dense_feature_dict(peplist, pep_uip, method_path, iupred2a)
    return x_pep.to(device), x_ss_pep.to(device), x_2_pep.to(device), x_diso_pep.to(device), x_bert_pep.to(device)