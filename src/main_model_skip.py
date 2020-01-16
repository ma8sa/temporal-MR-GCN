import os
import pandas as pd
import dgl
from dgl import DGLGraph
import torch
from random import shuffle
import time
import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
import dgl.function as fn
from functools import partial
import matplotlib.pyplot as plt
from random import sample
import tqdm
# from .activation import MultiheadAttention
# sys.path.append('./src/')
import RGCN_layer_skip_add
from RGCN_layer_skip_add import *



seed = 2
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def _get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    else:
        raise RuntimeError("activation should be relu/gelu, not %s." % activation)

# MY MODULE for MODEL and its parameters
class Model(nn.Module):
    def __init__(self, num_nodes, h_dim, out_dim, num_rels,h_dim2,h_dim3,h_dim4,dropout,skip=False,Fusion=False,
                 num_bases=-1,num_hidden_layers=1,use_cuda=False,gated=False):
        super(Model, self).__init__()
        self.num_nodes = num_nodes
        self.h_dim = h_dim
        self.out_dim = out_dim
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.num_hidden_layers = num_hidden_layers
        self.h_dim2 = h_dim2
        self.h_dim3 = h_dim3
        self.h_dim4 = h_dim4
        self.memory = []
        self.gated=gated
        self.use_cuda=use_cuda
        self.dropout = dropout
        self.skip = skip
        self.Fusion = Fusion
        # create rgcn layers
        self.build_model()

    def build_model(self):
        self.layers = nn.ModuleList()
        
        # input to hidden
        # i2h = self.build_input_layer()
        i2h = self.build_embedding_layer()
        self.layers.append(i2h)
        
        # hidden to hidden
        for _ in range(self.num_hidden_layers-1):
            h2h = self.build_hidden_layer()
            self.layers.append(h2h)
        
        self.layers.append(self.build_hidden_layer_final())
        self.layers.append(self.build_hidden_layer_final2())

        # h2o = self.build_output_layer()
        # self.layers.append(h2o)
        
    # def build_input_layer(self):
    #     return RGCNLayer(self.num_nodes, self.h_dim, self.num_rels,0,self.dropout ,self.num_bases,use_cuda=False,
    #                      activation=F.relu, is_input_layer=True,gated=self.gated)

    def build_embedding_layer(self):
        return Embed_Layer(self.num_nodes, self.h_dim,activation=F.relu, use_cuda=self.use_cuda)

    def build_hidden_layer(self):
        return RGCNLayer(self.h_dim, self.h_dim2, self.num_rels,1,self.skip,self.dropout,self.num_bases,use_cuda=self.use_cuda,
                         activation=F.relu,gated=self.gated)
    def build_hidden_layer_final(self):
        return RGCNLayer(self.h_dim2, self.h_dim3, self.num_rels,self.h_dim,self.skip,self.dropout ,self.num_bases,use_cuda=self.use_cuda,
                         activation=F.relu,gated=self.gated)
    
    def build_hidden_layer_final2(self):
        return RGCNLayer(self.h_dim3, self.h_dim4, self.num_rels,self.h_dim2,self.skip,self.dropout ,self.num_bases,use_cuda=self.use_cuda,
                         activation=F.relu,gated=self.gated,Fusion=self.Fusion)

    def build_output_layer(self):
        return RGCNLayer(self.h_dim2, self.out_dim, self.num_rels,1,self.skip,self.dropout,self.num_bases,use_cuda=self.use_cuda,
                         activation=partial(F.softmax, dim=1),gated=self.gated)

    def forward(self, g):
        self.hps = []
        for i in range(len(self.layers)):
            conv = self.layers[i]
            if(i>1):
                h,w = conv(g,i,self.hps[i-2]['h'],self.hps)
            else:
                h,w = conv(g,i,np.ones(g.number_of_nodes()),self.hps)

            self.hps.append({'h':h,'w':w})
        return g.ndata.pop('h')
        
class main_model(nn.Module):
    def __init__(self,num_node_fts,n_hidden,num_classes,num_rels,h_dim2,h_dim3,h_dim4,h_dim5,dropout,n_bases,n_hidden_layers,layers_lstm,dropout_lstm,time_stamps=10,use_cuda=False,bidirectional=True,gated=False,skip=False,Fusion=False):
        super(main_model, self).__init__()        
        self.num_node_fts = num_node_fts
        self.n_hidden=n_hidden
        self.num_classes=num_classes
        self.num_rels=num_rels
        self.h_dim2=h_dim2
        self.dropout=dropout
        self.n_bases=n_bases
        self.n_hidden_layers=n_hidden_layers
        self.h_dim3=h_dim3
        self.h_dim4=h_dim4
        self.h_dim5=h_dim5
        self.layers_lstm=layers_lstm
        self.dropout_lstm=dropout_lstm
        self.use_cuda=use_cuda
        self.bidirectional=bidirectional
        self.gated=gated
        self.skip=skip
        self.Fusion = Fusion
        print("SKIP_STATUS :",self.skip)
        print("GATED_STATUS : ",self.gated)
        print("FUSION_STATUS : ",self.Fusion)
        #Definfning each layer with given parameters
        self.rgcn = Model(self.num_node_fts,
                      self.n_hidden,
                      self.num_classes,
                      self.num_rels,self.h_dim2,self.h_dim3,self.h_dim4,
                      self.dropout,
                      self.skip,
                      self.Fusion,
                      num_bases=self.n_bases,
                      num_hidden_layers=self.n_hidden_layers,use_cuda=False,gated=self.gated)
     
        self.lstm = torch.nn.LSTM(input_size = self.h_dim4, hidden_size = self.h_dim5, num_layers = self.layers_lstm ,dropout = self.dropout_lstm ,bidirectional = self.bidirectional,batch_first=True)
        # self.lin_layer=nn.Linear(2*h_dim3,2*h_dim3)
        
        self.transformer = nn.TransformerEncoderLayer(2*h_dim5,1,16)
        self.pool = torch.nn.AvgPool1d(time_stamps)
        self.final_layer=nn.Linear(2*h_dim5,num_classes)
        # self.final_act = F.relu
        self.norm0 = nn.LayerNorm([time_stamps,h_dim3])
        self.norm1 = nn.LayerNorm([time_stamps,2*h_dim5])

        #only if cuda on
        if(self.use_cuda):
            self.rgcn.cuda()
            self.lstm.cuda()
            self.transformer.cuda()
            self.final_layer.cuda()
            # self.lin_layer.cuda()
            self.pool.cuda()
            self.norm1.cuda() 
            self.norm0.cuda() 
            # self.final_act.cuda()

    def forward_rgcn(self,trainsets,time_stamps,batch_size,j,whts,skipped,skipped_val,flag):
        
        k=j
        labels2 = []
        logits5 = []
        nodefts = []
        while(k<j+batch_size):
            check = 0
            if(k>=len(trainsets)):
                break
            trainset = trainsets[k][0]
            # print(trainset)
            # e= trainsets[k][1]

            gcn_ops=[]
            num_nodes = (trainset.graphs[0]).number_of_nodes()
            # some times needs an increase in number of nodes causes memory issue, so removing 
            # few graphs(at max skipping 4)
            # if(flag==1):
            #     print(num_nodes,j,flag)
            if(num_nodes>=90):
                if(flag==0):
                    skipped += 1
                    k += 1
                    check = 1
                    continue
                
                elif(flag==1):
                    skipped_val += 1
                    k += 1
                    check = 1
                    continue
            
            for t in range(time_stamps):    
                g = trainset.graphs[t]
                nodes_fts = g.ndata['id'].to('cpu')
                labs = trainset.labels[t]
                logits = self.rgcn(g)
                gcn_ops.append(logits)

            memory = torch.cat(gcn_ops ,dim=1)
            memory2 = torch.reshape(memory, (num_nodes,time_stamps,self.h_dim3))

            memory2 = self.norm0(memory2)
            logits2,hidden_op = self.lstm(memory2)
            logits2 = self.norm1(logits2)
            logits2 = F.relu(logits2)

            # now in n x t x d
            logits2 = logits2.permute(1,0,2)
            # now in t x n x d
            logits3 = self.transformer(logits2)
            # now in t x n x d
            logits3 = logits3.permute(1,0,2)
            # now in n x t x d
            logits3 = self.norm1(logits3)
            logits3 = logits3.permute(0,2,1)
            # now in n x d x t
            
            # logits3 = logits2.permute(0,2,1)
            logits3 = self.pool(logits3)
            logits3 = logits3.squeeze(dim=2)
            # Now in n x d

            # logits3 = F.dropout(logits2,p=0.2)
            logitsn = self.final_layer(logits3)
            logits4 = logitsn
            
            logits5.extend(logits4)
            labels2.extend(labs)
            nodefts.extend(nodes_fts)

            k += 1
        if(check==0):
            logits6 = torch.stack(logits5,dim=0)
            labels3 = torch.stack(labels2,dim=0)
            
            loss_func = nn.CrossEntropyLoss(weight=whts)
            loss2 = loss_func(logits6,labels3)
            return [loss2,logits6,labels3,nodefts,skipped,skipped_val]
        else:
            return [0,[],[],[],skipped,skipped_val]
