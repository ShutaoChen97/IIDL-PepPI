# -*- coding: utf-8 -*-
import numpy as np
import sys, os, subprocess
sys.path.append('Utility')
from multiprocessing import Pool
complet_n = 0

def run_simple_search(fd):
    protein_name = fd.split('.')[0]
    global complet_n
    complet_n += 1
    print('PSSM Processing:%s---%d' % (protein_name, complet_n*1))
    outfmt_type = 5
    num_iter = 10
    evalue_threshold = 0.001
    fasta_file = os.path.join(Profile_HOME, str(protein_name + '.fasta'))
    pssm_file = os.path.join(Profile_HOME, 'tmp', str( protein_name + '.pssm'))
    if os.path.isfile(pssm_file):
        pass
    else:
        cmd = ' '.join([BLAST,
                        '-query ' + fasta_file,
                        '-db ' + BLAST_DB,
                        '-evalue ' + str(evalue_threshold),
                        '-num_iterations ' + str(num_iter),
                        '-outfmt ' + str(outfmt_type),
                        '-out_ascii_pssm ' + pssm_file,  # Write the pssm file
                        '-num_threads ' + '1']
                       )
        subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL)

def generateMSA(file_path):
    seq_DIR = [file_path.rsplit('/', 1)[-1]]
    pssm_dir = Profile_HOME

    pool = Pool(1)
    results = pool.map(run_simple_search, seq_DIR)
    pool.close()
    pool.join()

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

def get_protein_blosum(method_path, proteinseq):
    blosum62 = {}
    blosum_reader = open(os.path.join(method_path, 'utils/blosum62.txt'), 'r')
    count = 0
    for line in blosum_reader:
        count = count + 1
        if count <= 7:
            continue
        line = line.strip('\r').split()
        blosum62[line[0]] = [float(x) for x in line[1:21]]
        
    protein_lst = []
    for aa in proteinseq:
        aa = aa.upper()
        if aa not in blosum62.keys():
            aa = 'X'
        protein_lst.append(blosum62[aa])
    return np.asarray(protein_lst)

def read_pssm(pssm_file):
    with open(pssm_file, 'r') as f:
        lines = f.readlines()
        lines = lines[3:-6]
        pro_seq = []
        mat = []
        for line in lines:
            tmp = line.strip('\n').split()
            pro_seq.append(tmp[1])
            tmp = tmp[2:22]
            mat.append(tmp)
        mat = np.array(mat)
        mat = mat.astype(float)
    return pro_seq, mat

def PSSM(method_path, ncbiblast, nrdb90, pro_uip):
    global BLAST
    global BLAST_DB
    BLAST = os.path.join(ncbiblast, 'bin/psiblast')
    BLAST_DB = os.path.join(nrdb90, 'nrdb90')

    file_path = pro_uip
    uip_tmp = pro_uip.rsplit('/', 1)
    if len(uip_tmp) == 2:
        uip = uip_tmp[0]
    
    global Profile_HOME
    Profile_HOME = uip
    generateMSA(file_path)
    
    _, prosequnique = Read_SeqID(pro_uip)
    
    protein_dense_feature_dict = {}
    for tmp in range(len(prosequnique)):
        proseqtmp = prosequnique[tmp]
        
        uip_tmp = pro_uip.rsplit('/', 1)
        if len(uip_tmp) == 2:
            uip = uip_tmp[0]
        
        if os.path.exists(os.path.join(uip_tmp[0], 'tmp', str(uip_tmp[1].split('.')[0] + '.pssm'))):
            pssmfeature = read_pssm(os.path.join(uip_tmp[0], 'tmp', str(uip_tmp[1].split('.')[0] + '.pssm')))[1]
        else:
            pssmfeature = get_protein_blosum(method_path, proseqtmp)
        protein_dense_feature_dict[prosequnique[tmp]] = pssmfeature
    return protein_dense_feature_dict
