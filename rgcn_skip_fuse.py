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

sys.path.append('./src/')
import main_model_skip
import graphs_preproc

from graphs_preproc import *
from main_model_skip import *


seed = 2
np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


if __name__ == '__main__':

    data_dir = '../lstm_graphs_apollo/'
    wts_dir = '/scratch/sravan/lstm_apollo_syntactic/'
    if(not(os.path.exists('/scratch/sravan/'))):
        os.mkdir('/scratch/sravan/')

    if(not(os.path.exists(wts_dir))):
        os.mkdir(wts_dir)

    use_cuda = 1
    if use_cuda:
            torch.cuda.set_device(0)

    ####################################################################
    #parameters for creating model 
    num_classes = 6
    n_hidden_layers = 2
    h_dim1 = 16          #rgcn 1st layer dimension
    h_dim2 = 16          #rgcn 2nd layer dimension
    h_dim3 = 16          #rgcn 3rd layer dimension
    h_dim4 = 16          #rgcn_o/p dimension
    h_dim5 = 8			 #LSTM input dimesnion	
    layers_lstm = 1
    dropout = 0.0           #dropout in rgcn
    dropout_lstm = 0.0
    
    num_node_fts = 2        #vehicle /static-points like lanes/poles; 0->vehicle, 1->lanes
    num_rels = 5            #relations are 4(top-left,bottom-left,top-right,bottom-right) and self-edge
    n_bases = -1
    ratio = 0.7

    print("data split",ratio)
    [trainsets,valsets,train_idx_nodes,count_class_train,count_class_val,count_overall_train] = create_data(num_classes,ratio,data_dir,2,use_cuda)
 
    #lenght of time steps for each sequence
    time_stamps = 10
    n_epochs = 100
    class_acc = [0.0]*num_classes
    
    skip = True
    Fusion = False
    model = main_model(num_node_fts,h_dim1,num_classes,num_rels,h_dim2,h_dim3,h_dim4,h_dim5,dropout,n_bases,n_hidden_layers,layers_lstm,dropout_lstm,use_cuda=use_cuda,bidirectional=True,gated=True,skip=skip,Fusion=Fusion)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001,weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',min_lr=0.00001,patience=2,verbose =True)
    
    print("train and val split")
    print("total training nodes = ",train_idx_nodes)
    print("train class counts ",count_class_train)
    print("val class counts ",count_class_val)
    print("overall class count on train",count_overall_train)
 
    whts=[]
    for i in range(num_classes):
        # whts.append(train_idx_nodes/count_overall_train[i])
        whts.append( ((train_idx_nodes-count_overall_train[i])/train_idx_nodes))
    
    # whts[0] *= 1.5
    # whts[1] *= 2.0
    # whts[3] *= 2.0
    # whts[5] *= 2.0

    whts = torch.from_numpy(np.array(whts))
    whts = whts.float()
    print(whts)
    print("created model")
    for param in model.parameters():
        print(param.shape)
        # torch.nn.init.xavier_uniform(param)
    loss_func = nn.CrossEntropyLoss(weight=whts)
    whts = whts.cuda()
    print("----------------------------------train and val split--------------------------------")
    print("train class counts ",count_class_train)
    print("val class counts ",count_class_val)
        
    print('\n\n')
    for epoch in range(n_epochs):
        st=time.time()
        train_crct = 0.0
        train_cntr = 1.0
        val_crct = 0.0
        val_cntr = 0.0
        train_loss = 0
        val_loss = 0
        val_loss_schdlr = 0
        
        train_class_crcts =  [0.0]*num_classes
        train_class_cnts =  [0.0]*num_classes
        val_class_crcts =  [0.0]*num_classes
        val_class_cnts =  [0.0]*num_classes
        skipped = 0
        skipped_val = 0

        tendency = [[0.0] * num_classes]*num_classes
        tendency = np.array(tendency)

        shuffle(trainsets)
        model.train()
        ##### TRAINING
        batch_size_train = 1
        for j in range(0,len(trainsets),batch_size_train):
            optimizer.zero_grad()
            loss2,logits6,labels3,nodefts,skipped,skipped_val = model.forward_rgcn(trainsets,time_stamps,batch_size_train,j,whts,skipped,skipped_val,0)
            if(loss2==0):
            	continue
            # print(loss2,'traininf')
            loss2.backward()
            optimizer.step()

            loss = loss2.to('cpu')
            del(loss2)
            train_loss += loss.item()
            logits7 = logits6.to('cpu')
            del(logits6)
            results = (logits7.argmax(dim=1))
            labels = labels3.to('cpu')
            del(labels3)
            torch.cuda.empty_cache()
        
            #For overall accuracies
            for h in range(len(results)):
                if(results[h].item()==labels[h].item() and nodefts[h].item() == 0 ):
                    train_crct += 1
                    train_cntr += 1
                elif(nodefts[h].item() == 0):
                    train_cntr += 1

            #For class accuracies
            for h in range(len(results)):
                if(results[h].item()==labels[h].item() and nodefts[h].item() == 0 ):
                    train_class_crcts[labels[h].item()] += 1
                    train_class_cnts[labels[h].item()] += 1
                elif(nodefts[h].item() == 0):
                    train_class_cnts[labels[h].item()] += 1

        torch.cuda.empty_cache()    
        train_acc = train_crct / train_cntr
        
        #VALIDATION
        model.eval()
        batch_size_val = 1
        for j in range(0,len(valsets),batch_size_val):
            loss2,logits6,labels3,nodefts,skipped,skipped_val = model.forward_rgcn(valsets,time_stamps,batch_size_val,j,whts,skipped,skipped_val,1)
            if(loss2==0):
            	continue
            # print(loss2,'validation')
            loss = loss2.to('cpu')
            del(loss2)
            val_loss += loss.item()
            val_loss_schdlr += loss.item()

            logits7 = logits6.to('cpu')
            del(logits6)

            results = (logits7.argmax(dim=1))
            labels = labels3.to('cpu')
            del(labels3)

            torch.cuda.empty_cache()    
            
            #For overall accuracies
            for h in range(len(results)):
                if(results[h].item()==labels[h].item() and nodefts[h].item() == 0 ):
                    val_crct += 1
                    val_cntr += 1
                elif(nodefts[h].item() == 0):
                    val_cntr += 1

            #For class accuracies
            for h in range(len(results)):
                if(results[h].item()==labels[h].item() and nodefts[h].item() == 0 ):
                    val_class_crcts[labels[h].item()] += 1
                    val_class_cnts[labels[h].item()] += 1
                elif(nodefts[h].item() == 0):
                    val_class_cnts[labels[h].item()] += 1

        scheduler.step(val_loss_schdlr)
        torch.cuda.empty_cache()

        val_acc = val_crct / val_cntr
        
        print("Epoch {:05d} | ".format(epoch) +
              "Train Accuracy: {:.4f} | Train Loss: {:.4f} | Val acc: {:.4f} | Val Loss: {:.4f} |".format(
                  train_acc, train_loss, val_acc, val_loss))
        print('----------------------------------------------------------------------------------------')

        train_class_crcts =np.array(train_class_crcts)
        train_class_cnts =np.array(train_class_cnts)

        val_class_crcts =np.array(val_class_crcts)
        val_class_cnts =np.array(val_class_cnts)

        train_cl_ac = train_class_crcts/train_class_cnts
        val_cl_ac   = val_class_crcts/val_class_cnts
        
        # if(val_acc>=0.5):
        #     best_acc=val_acc
        #     best_model = copy.deepcopy(model)
        #     checkpoint1 = {'model': best_model,
        #               'state_dict': best_model.state_dict(),
        #               'optimizer' : optimizer.state_dict()}

        #     torch.save(checkpoint1, wts_dir+'best_model_'+str(epoch)+'_'+str(best_acc)+'_'+str(val_cl_ac[0])+'_'+str(val_cl_ac[1])+'_'+str(val_cl_ac[2])+'.pth')
            
        #     for f in range(num_classes):
        #         if(class_acc[f] < val_cl_ac[f]):
        #             class_acc[f] = val_cl_ac[f]


        print("cl_ac",train_cl_ac,val_cl_ac)
        print(train_class_cnts,val_class_cnts)
        end=time.time()
        print("epoch time=",end-st)

    ########################## END OF CODE ##########################