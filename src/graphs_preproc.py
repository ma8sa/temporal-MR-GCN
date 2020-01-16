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


seed = 2
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False



class MiniGCDataset(object):

    def __init__(self, num_graphs):
        super(MiniGCDataset, self).__init__()
        self.num_graphs = num_graphs
        self.graphs = []
        self.labels = []

    def __len__(self):
        """Return the number of graphs in the dataset."""
        return len(self.graphs)

    def __add__(self,g,l):
        self.graphs.append(g)
        self.labels.append(l)
        
    def __getitem__(self, idx):
      
        return self.graphs[idx], self.labels[idx]


def create_g(file_path,use_cuda=False):
    # print(file_path)
    npz=np.load(file_path,allow_pickle=True)
    labels=npz['labels']
    fts_nodes=npz['fts_node']
    edge_type=npz['edge_type'].tolist()
    edge_norm=npz['edge_norm'].tolist()
    edges=npz['edges']

    
    num_nodes=int(npz['nums'])
    labels = labels[0:num_nodes]
    fts_nodes = fts_nodes[0:num_nodes]
    g = DGLGraph()
    g.add_nodes(num_nodes)

    edge_type=np.array(edge_type)
    edge_norm=np.array(edge_norm)
    # print(edges.shape)
    g.add_edges(edges[:,0],edges[:,1])
    
    edge_type = torch.from_numpy(edge_type)
    edge_norm = torch.from_numpy(edge_norm).unsqueeze(1)
    edge_norm = edge_norm.float()

    fts_nodes = fts_nodes.astype(int)
    fts_nodes = torch.from_numpy(fts_nodes)

    labels = torch.from_numpy(labels)
    
    if(use_cuda):
        labels = labels.cuda()
        edge_type = edge_type.cuda()
        edge_norm = edge_norm.cuda()
        fts_nodes = fts_nodes.cuda()
    
    g.edata.update({'rel_type': edge_type, 'norm': edge_norm})
    g.ndata['id']=fts_nodes
    return [g,labels,fts_nodes]


def no_labs(labels,req_class):
    for i in labels:
        if(i.item()==req_class):
            return False
    return True

def only2(labels,seq_file):
    cnt = 0.0

    for i in labels:
        if(i.item()==2):
            cnt += 1

    if(cnt/len(labels) >= 0.85 and no_labs(labels,3) and no_labs(labels,4) and  no_labs(labels,5) ):
        return True
    else:
        return False

def only0(labels,seq_file):
    cnt = 0.0

    for i in labels:
        if(i.item()==0):
            cnt += 1

    if(cnt/len(labels) >= 0.8 and no_labs(labels,3) and no_labs(labels,4) and no_labs(labels,5)):
        return True
    else:
        return False


def shuffle_seq(seqs,ratio):
    seqs = sorted(seqs)
    chunks = []
    for i in range(0,len(seqs),10):
        if(i+10<len(seqs)):
            # print(i,i+10)
            chunks.append([seqs[i:i+10]])
        else:
            chunks.append([seqs[i:len(seqs)]])
    ln=len(chunks)
    train_seq = random.sample(list(range(ln)), int(ratio * ln) )
    val_seq = []
    for i in range(len(chunks)):
        if(i not in train_seq):
            val_seq.append(i)

    train_list = []
    val_list = []
    print(len(train_seq),len(val_seq))
    for i in train_seq:
        for j in chunks[i][0]:
            train_list.append(j)
    for i in val_seq:
        for j in chunks[i][0]:
            val_list.append(j)

    print("train and val list lenhgths ",len(train_list),len(val_list))
    return [train_list,val_list]

def seperate_seq(graphs,ratio):

    print('------------------------seperate seqs splitting------------------')
    graphs = sorted(graphs)
    seqs_list = {}
    #createing train list and val list ; since graphs are of form seq_image.npz
    
    for i in graphs:
        seqs_list[ i.split('_')[0] ]=0
    seqs_list = [ v for v in seqs_list.keys() ]
    seqs_list = sorted(seqs_list)
    # print(seqs_list)
    
    train_list = []
    val_list = []
    val_seqs = []
    # print(len(seqs_list))
    
    # train_seqs = random.sample(list(range(len(seqs_list))), int(ratio * len(seqs_list)) )
    train_seqs = sample(seqs_list,int(ratio*len(seqs_list)))

    for i in (seqs_list):
        if(i in train_seqs):
            continue
        val_seqs.append(i)
    
    #running random and putting sequences here so that it is in sync with Tedge/for unedited set
    #for original data on apollo

    #------------------- unnecessary data splits. used only for prev experiments-----------------
    
    # train_seqs  = ['0064', '0022', '0029', '0051', '0001', '0039', '0021', '0006', '0015', '0011', '0030', '0040', '0034', '0072', '0017', '0023', '0055', '0024', '0061', '0000', '0010', '0018', '0013', '0009', '0036', '0026', '0002', '0008', '0014', '0019']
    # val_seqs = ['0016', '0027', '0028', '0031', '0032', '0033', '0044', '0046', '0052', '0060', '0062', '0066', '0070']
    
    #running random and putting sequences here so that it is in sync with Tedge/for unedited set
    #for edited_lane_chang
    # train_seqs = ['0064', '0022', '0029', '0051', '0001', '0039', '0021', '0006', '0015', '0011', '0030', '0040', '0034', '0072', '0017', '0023', '0055', '0024', '0061', '0000', '0010', '0018', '0013', '0009', '0036', '0026', '0002', '0008', '0014', '0019']
    # val_seqs = ['0016', '0027', '0028', '0031', '0032', '0033', '0044', '0046', '0052', '0060', '0062', '0066', '0070']

    #for depth data includeing all seqs 
    # train_seqs = ['0025', '0035', '0021', '0016', '0048', '0002', '0017', '0069', '0032', '0027', '0026', '0028', '0015', '0001', '0045', '0038', '0064', '0039', '0041', '0007', '0031', '0030', '0011']
    # val_seqs = ['0003', '0004', '0006', '0010', '0014', '0022', '0024', '0029', '0034', '0042', '0068', '0070', '0071','0013', '0037', '0033']

    #for depth apollo but includes flipped seqs(101-171)(old and not so crct ones)
    # train_seqs = ['123', '128', '0010', '0048', '156', '142', '125', '0069', '141', '0041', '0031', '0068', '135', '0022', '0045', '0014', '0027', '131', '121', '0011', '0070', '0025', '0056', '107', '145', '0015', '0028', '148', '0007', '0030', '157', '0057', '103', '139', '0037', '0029', '137', '0071', '0038', '171', '0003', '0034', '0026', '0002', '0004', '0032', '0021', '138', '101']
    # val_seqs = ['0006', '0013', '0016', '0017', '0023', '0024', '0033', '0035', '0039', '0042', '0064', '102', '106', '111', '114', '124', '126', '130', '133', '168', '169', '170']
    #------------------- unnecessary data splits. used only for prev experiments-----------------

    #actual data on which I got results
    train_seqs = ['0169', '0056','0104', '0068','0157','0138', '0022' ,'0118', '0110','0164','0026', '0124','0040', '0004', '0168','0123', '0050', '0018', '0137', '0116', '0149', '0115', '0139', '0061', '0141', '0014', '0016', '0146', '0003', '0156', '0034', '0070', '0023', '0002', '0031', '0010', '0043', '0015', '0038', '0055', '0134', '0142', '0037', '0148', '0064', '0006', '0111', '0041', '0125',
                '0025', '0107', '0045', '0028', '0011', '0013', '0102', '0069', '0143', '0170', '0042', '0032',
                'a_0002', 'a_0011', 'a_0014', 'a_0015',  'a_0051', 'a_0056', 'a_0057', 'a_0064', 'a_0069', 'a_0070', 'a_0102', 'a_0103', 'a_0111', 'a_0114', 'a_0115', 'a_0116', 'a_0124', 'a_0127', 'a_0137', 'a_0141', 'a_0143', 'a_0146', 'a_0148', 'a_0149', 'a_0151', 'a_0156', 'a_0157', 'a_0164', 'a_0168', 'a_0169', 'a_0170', 'a_0171','200','201','202','203','204','206','207','208','209',
                'a_0016', 'a_0024', 'a_0027', 'a_0037', 'a_0041', 'a_0042', 'a_0046', 'a_0048', 'a_0049',
                '300','301','302','303','304','306','307','308','309'
                ]

    val_seqs = ['0114', '0101', '0046', '0033', '0103', '0171', '0024', '0007', '0106', '0027', '0035', '0030', '0001', '0051', '0151', '0127', '0131', '0039', '0049', '0145', '0048', '0135', '0128',
                 '0017', '0057',  '0029', '0130'         
                ]
    #0.3 split data shuffled randomly
    # train_seqs = ['115', '124', '0010', '0011', '0014', '0016', '0018', '0024', '0027', '0031', '0037', '0007', '0056', '142', '138',
    #                '206', '0023', '200', '0051', '146', '170', '0030', '0071','111', '114', '123', '125',
    #               '202', '0015', '103', '135', '209', '300', '127', '101', '0042', '301', '128', '307', '0004', '303', '131', '0064', '151', '0001', '308', '110', '309',
    #               '0041', '148', '116', '156', '0006', '149', '204', '0025', '106', '157', '0057', '0045', '207', 'a', '104', '171', '304']
   
    # val_seqs = ['0002', '0003',  '0038', '0039', '0046', '0048', '0069', '102',  '134', '139', '143', '169', '201', '203', '208', '302', '306',
    #           '118', '0070', '137', '107', '164', '0043', '141', '0028', '0068', '145', '130', '168','0049', '0035', '0026', '0034']


    #shuffled again to see if network is able to learn clearly
    # train_seqs = ['169', '0056','104', '0068','157','138', '0022' ,'118', '110','164','0026', '124','0040', '0004', '168','123', '0050', '0018', '137', '116', '149', '115', '139', '0061', '141', '0014', '0016', '146', '0003', '156', '0034', '0070', '0023', '0002', '0031', '0010', '0043', '0015', '0038', '0055', '134', '142', '0037', '148', '0064', '0006', '111', '0041', '125', '0025', '107', '0045', '0028', '0011', '0013', '102', '0069', '143', '170', '0042', '0032',
    #             'a_0002', 'a_0011', 'a_0014', 'a_0015',  'a_0051', 'a_0056', 'a_0057', 'a_0064', 'a_0069', 'a_0070', 'a_102', 'a_103', 'a_111', 'a_114', 'a_115', 'a_116', 'a_124', 'a_127', 'a_137', 'a_141', 'a_143', 'a_146', 'a_148', 'a_149', 'a_151', 'a_156', 'a_157', 'a_164', 'a_168', 'a_169', 'a_170', 'a_171','200','201','202','203','204','206','207','208','209',
    #             'a_0016', 'a_0024', 'a_0027', 'a_0037', 'a_0041', 'a_0042', 'a_0046', 'a_0048', 'a_0049',
    #             '300','301','302','303','304','306','307','308','309'
    #             ]

    # val_seqs = ['114', '101', '0046', '0033', '103', '171', '0024', '0007', '106', '0027', '0035', '0030', '0001', '0051', '151', '127', '131', '0039', '0049', '145', '0048', '135', '128',
    #              '0017', '0057',  '0029', '130'         
    # ]



    print(train_seqs)
    print(val_seqs)
    print(len(train_seqs))
    print(len(val_seqs))

    for i in train_seqs:
        # print(seqs_list[i])
        for j in graphs:
            # print(j.split('_')[0])
            if((i == j.split('_')[0]) or (i == j.split('_')[0]+'_'+j.split('_')[1] ) ):
                train_list.append(j)


    for i in val_seqs:
        for j in graphs:
            if(i == j.split('_')[0] or (i == j.split('_')[0]+'_'+j.split('_')[1] ) ):
                val_list.append(j)

    print("train and val list lenhgths ",len(train_list),len(val_list))

    for i in train_list:
        if(i in val_list):
            print('biscuit')

    # for i in graphs:
    #     if(not( (i in train_list) or (i in val_list)  )):
    #         print(i)


    # print(train_list,val_list)
    return [train_list,val_list]

def get_random_seq(files,ratio):
    num_seq = len(files)
    val_list = []
    train_list = []
    val_len = int(num_seq*(1-ratio))

    for i in range(val_len):
        val_list.append(files[i])

    for i in range(val_len,num_seq):
        train_list.append(files[i])
    shuffle(train_list)
    shuffle(val_list)
    
    return [train_list,val_list]

def create_data(num_classes,ratio,data_path='./lstm_data/',split_meth=1,use_cuda=False,time_stamps=10):
    trainsets = []
    valsets = []
    files_dir = data_path
    seq_files = sorted(os.listdir(files_dir))
    # seq_files = seq_files[0:100]
    num_seq = len(seq_files)
    train_idx_nodes = 0
    count_class_train = [1.0] * num_classes
    count_class_val = [1.0] * num_classes
    count_overall_train = [1.0] * num_classes

    # train_list = random.sample(list(range(num_seq)), int(0.7 * num_seq) )
    # val_list = list(range(int(num_seq*0.3)))
    # train_list=[]
    # for i in range(num_seq):
    #     if(i not in val_list):
    #         train_list.append(i)

    if(split_meth==1):
        [ train_list,val_list ] = shuffle_seq(seq_files,0.8)
    else:
        [train_list,val_list] = seperate_seq(seq_files,ratio)
   
    # [ train_list,val_list ] = get_random_seq(seq_files,0.7)
    f=open('./train_list.txt','w')
    for i in train_list:
        f.write(i+'\n')

    f=open('./val_list.txt','w')
    for i in val_list:
        f.write(i+'\n')

    # print("data split",ratio)
    # print("break point ",None)          #(corresponding to value in function: only2)

    # train_list = train_list[0:10]
    # val_list = val_list[0:10]

    print("train list ",len(train_list))
    print("val list ",len(val_list))
    augment_1 = []
    augment_3 = []
    augment_5 = []
    
    #TRAIN DATA
    for j in tqdm.tqdm(train_list):
        graph_files = sorted(os.listdir(files_dir+j ))
        trainset = MiniGCDataset(time_stamps)
        flag = 0
        for i in graph_files:
            [g_curr,l_curr,node_features]=create_g(files_dir + j + '/' + i,use_cuda)
            test = only2(l_curr,j)
            if(test):
                flag = 1
                break
            # test = only0(l_curr,j)
            # if(test):
            #     flag = 1
            #     break
            train_idx_nodes += g_curr.number_of_nodes()
            trainset.__add__(g_curr,l_curr)
            for m in range(len(l_curr)):
                count_overall_train[l_curr[m].item()] += 1

        if(not(no_labs(l_curr,1))):
            augment_1.append(j)
        if(not(no_labs(l_curr,3))):
            augment_3.append(j)
        if(not(no_labs(l_curr,5))):
            augment_5.append(j)
        if(flag==0):
            for m in range(len(l_curr)):
                if(node_features[m].item()==0):
                    count_class_train[l_curr[m].item()] += 1 
            trainsets.append([trainset,j])

    print("initial",len(trainsets))
    print(len(augment_1),len(augment_3),len(augment_5))

    # for augmentatyion of class 3
    # shuffle(augment_1)
    # for j in augment_1:
    #     graph_files = sorted(os.listdir(files_dir+j))
    #     trainset = MiniGCDataset(time_stamps)
    #     for i in graph_files:
    #         [g_curr,l_curr,node_features]=create_g(files_dir + j + '/' + i,use_cuda)
    #         train_idx_nodes += g_curr.number_of_nodes()
    #         trainset.__add__(g_curr,l_curr)
    #         for m in range(len(l_curr)):
    #             count_overall_train[l_curr[m].item()] += 1
    #     for m in range(len(l_curr)):
    #         if(node_features[m].item()==0):
    #             count_class_train[l_curr[m].item()] += 1 
    #     trainsets.append([trainset,j])

    # shuffle(trainsets)
    # shuffle(trainsets)

    # # # for augmentatyion of class 0
    shuffle(augment_5)
    ag = 0
    while(ag<0):
        for j in augment_5:
            graph_files = sorted(os.listdir(files_dir+j))
            trainset = MiniGCDataset(len(graph_files))
            for i in graph_files:
                [g_curr,l_curr,node_features]=create_g(files_dir + j + '/' + i,use_cuda)
                train_idx_nodes += g_curr.number_of_nodes()
                trainset.__add__(g_curr,l_curr)
                for m in range(len(l_curr)):
                    count_overall_train[l_curr[m].item()] += 1
            for m in range(len(l_curr)):
                if(node_features[m].item()==0):
                    count_class_train[l_curr[m].item()] += 1 
            trainsets.append([trainset,j])
    
        shuffle(trainsets)
        shuffle(trainsets)
        ag += 1
    shuffle(trainsets)


    print("updated",len(trainsets))
    #VALIDATION DATA
    for j in tqdm.tqdm(val_list):
        graph_files = sorted(os.listdir(files_dir+j))
        valset = MiniGCDataset(time_stamps)
        for i in graph_files:
            [g_curr,l_curr,node_features]=create_g(files_dir + j + '/' + i,use_cuda)
            valset.__add__(g_curr,l_curr)
        for m in range(len(l_curr)):
            if(node_features[m].item()==0):
                count_class_val[l_curr[m].item()] += 1 
        valsets.append([valset,j])

    print("val len",len(valsets))
    shuffle(valsets)
    return [trainsets,valsets,train_idx_nodes,count_class_train,count_class_val,count_overall_train]

