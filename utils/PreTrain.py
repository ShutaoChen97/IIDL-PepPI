# %%
import os
import re
import torch
import pickle
import numpy as np
from sklearn.decomposition import PCA
from transformers import BertModel, BertTokenizer

# %%
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

# %%
# Scripts for extracting sequence word embeddings using ProtBERT pLM
def PreTrain(peplist, pep_uip, lenuse, method_path, protbert_path):
    tokenizer = BertTokenizer.from_pretrained(protbert_path, do_lower_case=False)
    model = BertModel.from_pretrained(protbert_path)

    _, sequnique = Read_SeqID(pep_uip)
    seqexa = []
    for tmp1 in range(len(sequnique)):
        seqtmp = sequnique[tmp1]
        sequse = ''
        count = 0
        for tmp2 in seqtmp:
            sequse += tmp2
            count += 1
            if count == len(seqtmp):
                continue
            sequse += ' '
        seqexa.append(sequse)

    bert_feature_dict = {}
    for tmp in range(len(seqexa)):
        # print('Current: ' + str(tmp) + '\n')
        seqlen = len(sequnique[tmp])
        sequence_Example = seqexa[tmp]
        sequence_Example = re.sub(r"[UZOB]", "X", sequence_Example)
        encoded_input = tokenizer(sequence_Example, return_tensors='pt')
        output = model(**encoded_input)
        result = output[0][0][1:seqlen+1]
        bert_feature = result.detach().numpy()
        if len(bert_feature) < lenuse:
            diff = lenuse - len(bert_feature)
            diffarr = np.zeros((diff, bert_feature.shape[1]))
            use = np.vstack((bert_feature, diffarr))
        else:
            use = bert_feature[0:lenuse, ]
        if use.shape[0] == lenuse:
            bert_feature_dict[sequnique[tmp]] = np.array(use)
        else:
            print('error!')
            break
    
    tmp = list(bert_feature_dict)[0]
    berttmp = bert_feature_dict[tmp][True, :]

    # PCA
    if lenuse == 50:
        with open(os.path.join(method_path, 'saved_models/protbert_feature_before_pca/peptide_webserver.pkl'), 'rb') as f:
            bert_use = pickle.load(f, encoding='iso-8859-1')
    else:
        with open(os.path.join(method_path, 'saved_models/protbert_feature_before_pca/protein_webserver.pkl'), 'rb') as f:
            bert_use = pickle.load(f, encoding='iso-8859-1')
            
    X = np.vstack((bert_use, berttmp))
    X_transform = np.zeros((X.shape[0], X.shape[1], 128))
    for i in range(X.shape[1]):
        pca = PCA(n_components=128)
        f  = pca.fit_transform(X[:, i, :])
        X_transform[:, i, :] = f
    
    bert_feature_dict[tmp] = X_transform[-1]
    out = Batch_Feature(peplist, bert_feature_dict, lenuse, 'float')
    return out.to(torch.float32)