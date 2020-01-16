#RCGCN START
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

class Embed_Layer(nn.Module):
    def __init__(self,input_dim,h_dim,activation=None,use_cuda=False):
        super(Embed_Layer, self).__init__()

        self.input_dim = input_dim
        self.h_dim = h_dim
        self.use_cuda = use_cuda
        self.embed = nn.Parameter(torch.Tensor(self.input_dim,self.h_dim))
        # self.embed = nn.Linear(self.input_dim-1,self.h_dim, bias=True)
        self.activation = activation 

    def forward(self, g,layer_num,h_skip,hps):
        #ids are 0/1 based on car/static-points. Hence only a 2xd matrix is the embedding matrix. 
        ids = g.ndata['id']
    
        # using a lookup table type-embedding
        h = self.embed[ids]
        
        # using a linear layer 1xd for learning the embedding
        # ids = ids.float()
        # h = self.embed(ids.unsqueeze(1))

        if self.activation:
            h = self.activation(h)
        
        g.ndata['h'] = h 
        return (g.ndata['h'],h_skip)

class RGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels,skip_dim,skip=False,dropout=0.0,num_bases=-1,use_cuda=False, bias=None,
                 activation=None, is_input_layer=False,gated=False,Fusion=False):
        super(RGCNLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.bias = bias
        self.activation = activation
        self.is_input_layer = is_input_layer
        self.use_cuda = use_cuda
        self.gated=gated
        self.skip_dim = skip_dim
        self.skip = skip
        self.Fusion = Fusion
        # self.dropout = dropout
        # sanity check
        if self.num_bases <= 0 or self.num_bases > self.num_rels:
            self.num_bases = self.num_rels

        # weight bases in equation (3)
        self.weight = nn.Parameter(torch.Tensor(self.num_bases, self.in_feat,
                                                self.out_feat))
        if self.num_bases < self.num_rels:
            # linear combination coefficients in equation (3)
            self.w_comp = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))

        # add bias
        if self.bias:
            self.bias = nn.Parameter(torch.Tensor(out_feat))

        if self.gated:
            self.bias_gate = nn.Parameter(torch.Tensor(1))

        # init trainable parameters
        nn.init.xavier_uniform_(self.weight,
                                gain=nn.init.calculate_gain('relu'))
        if self.num_bases < self.num_rels:
            nn.init.xavier_uniform_(self.w_comp,
                                    gain=nn.init.calculate_gain('relu'))
        if self.bias:
            nn.init.xavier_uniform_(self.bias,
                                    gain=nn.init.calculate_gain('relu'))
        if self.gated:
            if self.is_input_layer:
                self.gate_weight = nn.Parameter(torch.Tensor(self.num_rels,1, 1))
            else:
                self.gate_weight = nn.Parameter(torch.Tensor(self.num_rels,self.in_feat, 1))
            
            nn.init.xavier_uniform_(self.gate_weight,gain=nn.init.calculate_gain('sigmoid'))
    
        self.skip_weight = nn.Linear(self.out_feat+self.skip_dim,self.out_feat, bias=True)
        # self.w_skip = nn.Parameter(torch.Tensor(self.out_feat+32,self.out_feat))

        if self.Fusion:
            Fusion_dim = 16+16+16
            self.Fusion_weight = nn.Linear(self.out_feat+Fusion_dim,self.out_feat, bias=True)

    def forward(self, g,layer_num,h_skip,hps):
        if self.num_bases < self.num_rels:
            # generate all weights from bases (equation (3))
            weight = self.weight.view(self.in_feat, self.num_bases, self.out_feat)
            weight = torch.matmul(self.w_comp, weight).view(self.num_rels,
                                                        self.in_feat, self.out_feat)
        else:
            weight = self.weight

        if self.gated:
            gate_weight = self.gate_weight
        
        if self.is_input_layer:
            def message_func(edges):
                # for input layer, matrix multiply can be converted to be
                # an embedding lookup using source node id
                embed = weight.view(-1, self.out_feat)
                index = edges.data['rel_type'] + self.in_feat * edges.src['id']
                # print(index,embed.shape,edges.data['rel_type'])
                msg   = embed[index] * edges.data['norm']

                if self.gated:
                    gate_w = gate_weight[edges.data['rel_type']]
                    # print(index.shape,gate_w.shape,self.in_feat,self.out_feat)
                    index = index.float()
                    gate = torch.bmm(index.unsqueeze(1).unsqueeze(2), gate_w).squeeze().reshape(-1,1)
                    gate = torch.sigmoid(gate)
                    msg = msg * gate
                return {'msg': msg}
        else:
            def message_func(edges):
                w = weight[edges.data['rel_type']]                
                msg = torch.bmm(edges.src['h'].unsqueeze(1), w).squeeze()
                msg = msg * edges.data['norm']
                
                if self.gated:
                    gate_w = gate_weight[edges.data['rel_type']]
                    gate = torch.bmm(edges.src['h'].unsqueeze(1), gate_w).squeeze().reshape(-1,1)
                    gate = gate+ self.bias_gate
                    gate = torch.sigmoid(gate)
                    msg = msg * gate
                return {'msg': msg}
        
        def apply_func(nodes):
            h = nodes.data['h']
            if self.bias:
                h = h + self.bias
            if self.activation:
                h = self.activation(h)

            #skip-connections from i-2 th layer to the current layer                        
            if(layer_num > 1 and self.skip):

                h = torch.cat([h,h_skip],1)
                h = self.skip_weight(h)
            
            if self.activation:
                h = self.activation(h)

            if self.Fusion:
                for i in range(len(hps)-1,-1,-1):
                    h=torch.cat([h,hps[i]['h']],1)
                h = self.Fusion_weight(h)

                if self.activation:
                    h = self.activation(h)

            return {'h': h}
        
        g.update_all(message_func, fn.sum(msg='msg', out='h'), apply_func)
        return (g.ndata['h'],weight)

        # g.update_all(message_func, fn.copy_src(src='msg', out='h'), apply_func)