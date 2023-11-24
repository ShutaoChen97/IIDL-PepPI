# -*- coding: utf-8 -*-
# %%
import math
import torch
from torch import nn
import warnings
warnings.filterwarnings("ignore")

# %%
class Self_Attention(nn.Module):
    # input : batch_size * seq_len * input_dim
    # q : batch_size * input_dim * dim_k
    # k : batch_size * input_dim * dim_k
    # v : batch_size * input_dim * dim_v
    def __init__(self,input_dim,dim_k,dim_v):
        super(Self_Attention,self).__init__()
        self.q = nn.Linear(input_dim,dim_k)
        self.k = nn.Linear(input_dim,dim_k)
        self.v = nn.Linear(input_dim,dim_v)
        self._norm_fact = 1 / math.sqrt(dim_k)
        
    
    def forward(self,x):
        Q = self.q(x) # Q: batch_size * seq_len * dim_k
        K = self.k(x) # K: batch_size * seq_len * dim_k
        V = self.v(x) # V: batch_size * seq_len * dim_v
         
        atten = nn.Softmax(dim=-1)(torch.bmm(Q,K.permute(0,2,1))) * self._norm_fact # Q * K.T() # batch_size * seq_len * seq_len
        
        output = torch.bmm(atten,V) # Q * K.T() * V # batch_size * seq_len * dim_v
        
        return output

class CNN(nn.Module):
    def __init__(self, input_dim, out_dim, kernel):
        super(CNN,self).__init__()
        self.conv1d = nn.Sequential(
            nn.Conv1d(in_channels=input_dim, 
                      out_channels=out_dim, 
                      kernel_size=kernel, 
                      stride=1, padding='same', 
                      bias=True),
            nn.BatchNorm1d(out_dim),
            nn.LeakyReLU(0.02)
        )
        
    def forward(self, x):
        output = self.conv1d(x)
        return output

class TextCNN(nn.Module):
    def __init__(self, input_dim, out_dim, kernel=[]):
        super(TextCNN,self).__init__()
        layer = []
        for i,os in enumerate(kernel):
            layer.append(CNN(input_dim, out_dim, os))
        self.layer = nn.ModuleList(layer)
    def forward(self, x):
        x = x.permute(0, 2, 1)
        o1 = self.layer[0](x).permute(0, 2, 1)
        o2 = self.layer[1](x).permute(0, 2, 1)
        o3 = self.layer[2](x).permute(0, 2, 1)
        o4 = self.layer[3](x).permute(0, 2, 1)
        return o1, o2, o3, o4

class ConvNN(nn.Module):
    def __init__(self,in_dim,c_dim,kernel_size):
        super(ConvNN,self).__init__()
        self.convs = nn.Sequential(
            nn.Conv1d(in_channels=in_dim, out_channels= c_dim, kernel_size=kernel_size,padding='same'),
            nn.ReLU(),
            nn.Conv1d(in_channels=c_dim, out_channels= c_dim*2, kernel_size=kernel_size,padding='same'),
            nn.ReLU(),
            nn.Conv1d(in_channels=c_dim*2, out_channels= c_dim*3, kernel_size=kernel_size,padding='same'),
            nn.ReLU(),
            )
    def forward(self,x):
        x = self.convs(x)
        return x

class Attention(nn.Module):
    def __init__(self, weight_dim, feature_dim, seq_len):
        super().__init__()
        self.w = nn.Parameter(torch.rand(feature_dim, weight_dim))
        self.b = nn.Parameter(torch.zeros(weight_dim))
        self.bn = nn.BatchNorm1d(seq_len)
        self.W_attention = nn.Linear(feature_dim, weight_dim)
        
    def forward(self, sum_input, weight_output):
        h = torch.relu(self.W_attention(sum_input))
        hs = torch.relu(self.W_attention(weight_output))
        weight = torch.sigmoid(torch.matmul(h, hs.permute(0, 2, 1))).permute(0, 2, 1)
        h_output = weight * hs
        return h_output, weight

class GlobalMaxPool1d(nn.Module):
    def __init__(self):
        super(GlobalMaxPool1d,self).__init__()
    def forward(self,x):
        output, _ = torch.max(x,1)
        return output

# %%
class IIDLPepPI(nn.Module):
    def __init__(self):
        super(IIDLPepPI, self).__init__()
        self.embed_seq = nn.Embedding(25,128)
        self.embed_ss = nn.Embedding(73,128)
        self.embed_two = nn.Embedding(8,128)
        self.dense_pep = nn.Linear(3,128)    
        self.dense_prot = nn.Linear(23,128)
        self.dense_bert_pep = nn.Linear(128, 128)
        self.dense_bert_pro = nn.Linear(128, 128)

        self.pep_convs = TextCNN(640, 64, [3,5,7,9])
        self.prot_convs = TextCNN(640, 64, [5,10,15,20])

        self.global_max_pooling = GlobalMaxPool1d()
        self.pep_residue = nn.Sequential(nn.Linear(192,1), nn.Sigmoid())

        self.peptopro = Attention(384, 384, 800)
        self.protopep = Attention(384, 384, 50)

        self.dnns = nn.Sequential(
            nn.Linear(768,1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024,1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024,512))
        
        self.att = Self_Attention(128,128,128)
        self.output = nn.Linear(512,1)

    def forward(self, x_pep, x_ss_pep, x_2_pep, x_diso_pep, x_bert_pep, 
                x_p, x_ss_p, x_2_p, x_dense_p, x_bert_p):
        
        # ================= Feature embedding =================
        pep_seq_emb = self.embed_seq(x_pep)
        prot_seq_emb = self.embed_seq(x_p)
        pep_ss_emb = self.embed_ss(x_ss_pep)
        prot_ss_emb = self.embed_ss(x_ss_p)
        pep_2_emb = self.embed_two(x_2_pep)
        prot_2_emb = self.embed_two(x_2_p)
        pep_dens1 = self.dense_pep(x_diso_pep)
        prot_dens1 = self.dense_prot(x_dense_p)
        pep_bert = self.dense_bert_pep(x_bert_pep)
        prot_bert = self.dense_bert_pro(x_bert_p)

        
        # =================Feature concatenate=================
        encode_peptide = torch.cat([pep_seq_emb, pep_ss_emb, 
                                    pep_2_emb, pep_dens1, pep_bert],dim=-1)
        encode_protein = torch.cat([prot_seq_emb, prot_ss_emb, 
                                    prot_2_emb, prot_dens1, prot_bert],dim=-1)
        
        
        # =================       TextCNN       =================
        # -------------------For protein-------------------
        c1_p, c2_p, c3_p, c4_p = self.prot_convs(encode_protein)
        encode_protein = torch.cat([c1_p, c2_p, c3_p, c4_p], dim=-1)
        # -------------------For peptide-------------------
        c1_pep, c2_pep, c3_pep, c4_pep = self.pep_convs(encode_peptide)
        encode_peptide = torch.cat([c1_pep, c2_pep, c3_pep, c4_pep], dim=-1)

        # =================Seld-Attention model=================
        # -------------------For protein-------------------
        prot_seq_att = self.embed_seq(x_p)
        protein_att = self.att(prot_seq_att)
        # -------------------For peptide-------------------
        pep_seq_att = self.embed_seq(x_pep)
        peptide_att = self.att(pep_seq_att)


        # =================Bi-Attention model=================
        feature_pep = torch.cat([encode_peptide, peptide_att], dim=-1)
        feature_p = torch.cat([encode_protein, protein_att], dim=-1)
        # -------------------For protein-------------------
        feature_pep_sum = self.global_max_pooling(feature_pep).unsqueeze(1)
        _, weight_peptop = self.peptopro(feature_pep_sum, feature_p)
        feature_p = feature_p * weight_peptop
        # -------------------For peptide-------------------
        feature_p_sum = self.global_max_pooling(feature_p).unsqueeze(1)
        _, weight_ptopep = self.protopep(feature_p_sum, feature_pep)
        feature_pep = feature_pep * weight_ptopep


        # =================Global max pooling=================
        glomax_pep = self.global_max_pooling(feature_pep)
        glomax_p = self.global_max_pooling(feature_p)


        # =================Feature concatenate=================
        encode_interaction = torch.cat([glomax_pep, glomax_p], dim=-1)

        
        # =================        MLP        =================
        encode_interaction = self.dnns(encode_interaction)
        predictions = torch.sigmoid(self.output(encode_interaction))
        return predictions


class IIDLPepPIRes(nn.Module):
    def __init__(self):
        super(IIDLPepPIRes, self).__init__()
        self.embed_seq = nn.Embedding(25,128)
        self.embed_ss = nn.Embedding(73,128)
        self.embed_two = nn.Embedding(8,128)
        self.dense_pep = nn.Linear(3,128)    
        self.dense_prot = nn.Linear(23,128)
        self.dense_bert_pep = nn.Linear(128, 128)
        self.dense_bert_pro = nn.Linear(128, 128)

        self.pep_convs = TextCNN(640, 64, [3,5,7,9])
        self.prot_convs = TextCNN(640, 64, [5,10,15,20])

        self.global_max_pooling = GlobalMaxPool1d()
        self.pep_residue = nn.Sequential(nn.Linear(192,1), nn.Sigmoid())

        self.peptopro = Attention(384, 384, 800)
        self.protopep = Attention(384, 384, 50)

        self.dnns = nn.Sequential(
            nn.Linear(850,1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024,1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024,1024))
        
        self.att = Self_Attention(128,128,128)
        self.output = nn.Linear(1024,850)

    def forward(self, x_pep, x_ss_pep, x_2_pep, x_diso_pep, x_bert_pep, 
                x_p, x_ss_p, x_2_p, x_dense_p, x_bert_p):
        
        pep_seq_emb = self.embed_seq(x_pep)
        prot_seq_emb = self.embed_seq(x_p)
        pep_ss_emb = self.embed_ss(x_ss_pep)
        prot_ss_emb = self.embed_ss(x_ss_p)
        pep_2_emb = self.embed_two(x_2_pep)
        prot_2_emb = self.embed_two(x_2_p)
        pep_dens1 = self.dense_pep(x_diso_pep)
        prot_dens1 = self.dense_prot(x_dense_p)
        pep_bert = self.dense_bert_pep(x_bert_pep)
        prot_bert = self.dense_bert_pro(x_bert_p)

        encode_peptide = torch.cat([pep_seq_emb, pep_ss_emb, 
                                    pep_2_emb, pep_dens1, pep_bert],dim=-1)
        encode_protein = torch.cat([prot_seq_emb, prot_ss_emb, 
                                    prot_2_emb, prot_dens1, prot_bert],dim=-1)
 
        c1_pep, c2_pep, c3_pep, c4_pep = self.pep_convs(encode_peptide)
        encode_peptide = torch.cat([c1_pep, c2_pep, c3_pep, c4_pep], dim=-1)
        c1_p, c2_p, c3_p, c4_p = self.prot_convs(encode_protein)
        encode_protein = torch.cat([c1_p, c2_p, c3_p, c4_p], dim=-1)

        pep_seq_att = self.embed_seq(x_pep)
        peptide_att = self.att(pep_seq_att)
        prot_seq_att = self.embed_seq(x_p)
        protein_att = self.att(prot_seq_att)

        feature_pep = torch.cat([encode_peptide, peptide_att], dim=-1)
        feature_p = torch.cat([encode_protein, protein_att], dim=-1)
        feature_pep_sum = self.global_max_pooling(feature_pep).unsqueeze(1)
        _, weight_peptop = self.peptopro(feature_pep_sum, feature_p)
        feature_p = feature_p * weight_peptop
        feature_p_sum = self.global_max_pooling(feature_p).unsqueeze(1)
        _, weight_ptopep = self.protopep(feature_p_sum, feature_pep)
        feature_pep = feature_pep * weight_ptopep
        glomax_pep, _ = torch.max(feature_pep, -1)
        glomax_p, _ = torch.max(feature_p, -1)
        
        encode_interaction = torch.cat([glomax_pep, glomax_p], dim=-1)
        encode_interaction = self.dnns(encode_interaction)
        predictions = torch.sigmoid(self.output(encode_interaction))
        return predictions
