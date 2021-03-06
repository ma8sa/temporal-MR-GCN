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
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
import dgl.function as fn
from functools import partial
from random import sample
import tqdm

from graphs_preproc import *
from graphs_preproc_kitti_indian import *
from rgcn_layer import *
from main_model import *

seed = 0
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if __name__ == '__main__':
    if(sys.argv[1]=='kitti'):
        data_dir = './lstm_graphs_kitti/'
    else:
        data_dir = './lstm_graphs_indian/'

    use_cuda = 0
    if use_cuda:
            torch.cuda.set_device(0)

    ####################################################################
    #parameters for creating model 
    num_classes = 6
    n_hidden_layers = 1
    n_hidden = 128          #rgcn 1st layer dimension
    h_dim2 = 64             #rgcn 2nd layer dimension
    h_dim3 = 32             # FOR LSTM o/p dimension
    layers_lstm = 1
    dropout = 0.0           #dropout in rgcn
    dropout_lstm = 0.0
    
    num_node_fts = 2        #vehicle /static-points like lanes/poles; 0->vehicle, 1->lanes
    num_rels = 5            #relations are 4(top-left,bottom-left,top-right,bottom-right and self-edge)
    n_bases = -1
    ratio = 0.0

    print("data split",ratio)
    if(sys.argv[1]=='kitti'):
        [trainsets,valsets,train_idx_nodes,count_class_train,count_class_val,count_overall_train] = create_data_kitti_indian(num_classes,ratio,'kitti',data_dir,2,use_cuda)
    else:
        [trainsets,valsets,train_idx_nodes,count_class_train,count_class_val,count_overall_train] = create_data_kitti_indian(num_classes,ratio,'indian',data_dir,2,use_cuda)

    #lenght of time steps for each sequence
    time_stamps = 10
    n_epochs = 1
    
    class_acc = [0.0]*num_classes
    model = main_model(num_node_fts,n_hidden,num_classes,num_rels,h_dim2,dropout,n_bases,n_hidden_layers,h_dim3,layers_lstm,dropout_lstm,use_cuda=use_cuda,bidirectional=True)
    model_path = './final_model.pth'
    model.load_state_dict(torch.load(model_path,map_location='cpu')['state_dict'])

    print("created model")
    for param in model.parameters():
        print(param.shape)
        # torch.nn.init.xavier_uniform(param)
    loss_func = nn.CrossEntropyLoss()

    print("----------------------------------train and val split--------------------------------")
    print("train class counts ",count_class_train)
    print("val class counts ",count_class_val)
        
    print('\n\n')
    
    for epoch in range(n_epochs):
        st=time.time()
        val_crct = 0.0
        val_cntr = 0.0
        val_loss = 0
        val_loss_schdlr = 0
        
        val_class_crcts =  [0.0]*num_classes
        val_class_cnts =  [1.0]*num_classes
        
        skipped = 0
        skipped_val = 0
        batch_size_train = 1

        #VALIDATION SET TESTING
        batch_size_val = 1
        model.eval()
        for j in tqdm.tqdm(range(0,len(valsets),batch_size_val)):
            loss2,logits6,labels3,nodefts,skipped,skipped_val = model.forward_rgcn(valsets,time_stamps,batch_size_val,j,skipped,skipped_val,1)
            loss = loss2.to('cpu')
            del(loss2)
            val_loss += loss.item()
            val_loss_schdlr += loss.item()

            logits7 = logits6.to('cpu')
            del(logits6)

            results = (logits7.argmax(dim=1))
            labels = labels3.to('cpu')
            for pos in range(len(results)):
                if (results[pos]==3 or results[pos]==4 or results[pos]==5):
                    results[pos] = 0
            del(labels3)

            torch.cuda.empty_cache()    
            for h in range(len(results)):
                if(results[h].item()==labels[h].item() and nodefts[h].item() == 0 ):
                    val_crct += 1
                    val_cntr += 1
                elif(nodefts[h].item() == 0):
                    val_cntr += 1

            for h in range(len(results)):
                if(results[h].item()==labels[h].item() and nodefts[h].item() == 0 ):
                    val_class_crcts[labels[h].item()] += 1
                    val_class_cnts[labels[h].item()] += 1
                elif(nodefts[h].item() == 0):
                    val_class_cnts[labels[h].item()] += 1

        torch.cuda.empty_cache()

        val_acc = val_crct / val_cntr
        
        print("Epoch {:05d} | ".format(epoch) +
              " Val acc: {:.4f} | Val Loss: {:.4f} |".format(
                 val_acc, val_loss))
        print('----------------------------------------------------------------------------------------')

        val_class_crcts =np.array(val_class_crcts)
        val_class_cnts =np.array(val_class_cnts)

        val_cl_ac   = val_class_crcts/val_class_cnts

        print("calss accuracies",val_cl_ac[0:3])
        print(val_class_cnts[0:3])
        end=time.time()
        print("epoch time=",end-st)

    ########################## END OF CODE ##########################
