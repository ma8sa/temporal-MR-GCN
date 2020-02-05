import os
import numpy as np
from sklearn.model_selection import train_test_split
import torch
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

# seed = 2
# np.random.seed(seed)
# torch.manual_seed(seed)
# random.seed(seed)

# torch.cuda.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = True


class main_model(nn.Module):
    def __init__(self,input_size,hidden_size,layers_lstm,num_classes,use_cuda=False,bidirectional=True):
        super(main_model, self).__init__()        
        time_stamps = 10
        self.use_cuda=use_cuda
        self.bidirectional=bidirectional
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.layers_lstm = layers_lstm
        self.lstm = torch.nn.LSTM(input_size = self.input_size, hidden_size = self.hidden_size,num_layers = self.layers_lstm ,bidirectional = self.bidirectional,batch_first=True)
        self.transformer = nn.TransformerEncoderLayer(2*hidden_size,16,1024)
        self.pool = torch.nn.AvgPool1d(time_stamps)
        self.final_layer=nn.Linear(2*hidden_size,num_classes)
        self.norm1 = nn.LayerNorm([time_stamps,2*hidden_size])

        #
        self.l1=nn.Linear(input_size*10,128)
        self.n1 = nn.LayerNorm(128)
        self.l2=nn.Linear(128,32)
        self.n2 = nn.LayerNorm(32)
        self.l3=nn.Linear(32,num_classes)
        self.l4=nn.Linear(2*self.hidden_size,num_classes)

        self.lstm_mlp = torch.nn.LSTM(input_size = self.input_size, hidden_size = self.hidden_size,num_layers = self.layers_lstm ,bidirectional = self.bidirectional,batch_first=True)
        self.l1_mlp = nn.Linear(2*hidden_size*time_stamps,64)
        self.n1_mlp = nn.LayerNorm(64)
        self.l2_mlp = nn.Linear(64,5)

        #only if cuda on
        if(self.use_cuda):
            self.lstm.cuda()
            self.final_layer.cuda()
            self.pool.cuda()
            self.norm1.cuda()
            self.transformer.cuda()

            self.l1.cuda()
            self.l2.cuda()
            self.l3.cuda()
            self.l4.cuda()
            self.n1.cuda()
            self.n2.cuda()

            self.lstm_mlp.cuda()
            self.l1_mlp.cuda()
            self.n1_mlp.cuda()
            self.l2_mlp.cuda()


    def forward_rgcn(self,X,Y):
            
    		# # UNOCMMENT For LSTM + Attetnion(L+MA) BASELINE
      #       logits2,hidden_op = self.lstm(X)
      #       logits2 = self.norm1(logits2)
      #       logits2 = F.relu(logits2)

      #       # to get into t x N x d format since transformers take that input
      #       logits2 = logits2.permute(1,0,2)
      #       logits3 = self.transformer(logits2)
            
      #       # Now in t x n x d format
      #       ####MAX pooling over time dimension
      #       logits3 = logits3.permute(1,0,2)
      #       # Now in n x t x d
      #       logits3 = logits3.permute(0,2,1)
      #       # Now in n x d x t
            
      #       logits3 = self.pool(logits3)
      #       logits3 = logits3.squeeze(dim=2)

      #       # Now in n x d
      #       logits4 = self.final_layer(logits3)
      #       return logits4

            #UNCOMMENT for LSTM BASELINE (L) 
            logits2,hidden_op = self.lstm(X)
            logits2 = self.norm1(logits2)
            logits2 = F.relu(logits2)
            logits3 = logits2.permute(0,2,1)
            # now in n x d x t
            logits3 = self.pool(logits3)
            logits3 = logits3.squeeze(dim=2)
                    
            op4 = self.l4(logits3)
            return op4

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

            # #uncomment for mlp
            # op1 = self.l1(X)
            # op1 = self.n1(op1)
            # op1 = F.relu(op1)

            # op2 = self.l2(op1)  
            # op2 = self.n2(op2)
            # op2 = F.relu(op2)

            # op3 = self.l3(op2)
            # return op3

            #uncomment for lstm+mlp
            # logits2,hidden_op = self.lstm_mlp(X)
            # logits2 = self.norm1(logits2)
            # logits2 = F.relu(logits2)
            # #now in n x t x d
            # logits2 = logits2.view(logits2.shape[0],-1)

            # op1 = self.l1_mlp(logits2)
            # op1 = self.n1_mlp(op1)
            # op1 = F.relu(op1)

            # op2 = self.l2_mlp(op1)  
            # return op2
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

def get_whts(Y,num_classes):
	total_cnt = 0.0
	cnt_class = [0.0] * num_classes
	for i in Y:
		total_cnt += 1
		cnt_class[i] += 1
	whts = [0.0] * num_classes

	for i in range(num_classes):
		whts[i] = (total_cnt - cnt_class[i])/total_cnt
	return whts

if __name__ == '__main__':

	X_train = np.load('features_train_svm.npy')
	X_test =  np.load('features_test_svm.npy')
	Y_train = np.load('labels_train_svm.npy') 
	Y_test =  np.load('labels_test_svm.npy')

	X_train = torch.from_numpy(X_train)
	X_test = torch.from_numpy(X_test)
	Y_train = torch.from_numpy(Y_train)	
	Y_test = torch.from_numpy(Y_test)	
	
	print(X_train.shape,Y_train.shape)
	print(X_test.shape,Y_test.shape)
	print("created data")
	
	# SVM base-line
	# whts = {0:1.5,1:2.0,2:1.0,3:2.0,4:2.0,5:4.0}
	# lin_clf = svm.SVC(gamma='scale',kernel='poly',verbose=True, class_weight=whts)#,max_iter=10)#'balanced')
	# lin_clf.fit(X_train,y_train)
	# print('done fitting model,predicting...')
	# total_predictions = lin_clf.predict(X_test)
	# total_labels = y_test
    
	total_epochs = 1000
	num_classes = 6

	#LSTM baseline
	layers_lstm = 1
	bidirectional = True
	input_size = X_train.shape[2]
	hidden_size = 128
	use_cuda = True
	if use_cuda:
		torch.cuda.set_device(0)
	model = main_model(input_size,hidden_size,layers_lstm ,num_classes,bidirectional=bidirectional,use_cuda=use_cuda)
	optimizer = torch.optim.Adam(model.parameters(), lr=0.001,weight_decay=1e-4)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',min_lr=0.00001,patience=4,verbose =True)       
	whts = get_whts(Y_train,num_classes)
	whts[-1] *= 2.0

	print(whts)
	batch_size_train = 100
	batch_size_val = 100
	if use_cuda:
		whts = torch.from_numpy(np.array(whts))
		whts = whts.cuda()
		whts = whts.float()
		loss_func = nn.CrossEntropyLoss(weight=whts)

	org_whts = copy.deepcopy(whts)
	for epoch in range(total_epochs):
		# whts = copy.deepcopy(org_whts)

		loss_func = nn.CrossEntropyLoss(weight=whts)

		st=time.time()
		train_crct = 0.0
		train_cntr = 0.0
		val_crct = 0.0
		val_cntr = 0.0
		train_loss = 0
		val_loss = 0
		val_loss_schdlr = 0

		train_class_crcts =  [0.0]*num_classes
		train_class_cnts =  [0.0]*num_classes
		val_class_crcts =  [0.0]*num_classes
		val_class_cnts =  [0.0]*num_classes

		#training 
		model.train()
		
		batches = X_train.shape[0]/batch_size_train
		if(not(X_train.shape[0]%batch_size_train == 0)):
			batches += 1
		batches = int(batches)

		for j in range(batches):
			curr_X = X_train[j*batch_size_train:min(X_train.shape[0],(j+1)*batch_size_train),:,:]
			curr_Y = Y_train[j*batch_size_train:min(X_train.shape[0],(j+1)*batch_size_train)]

			curr_Y = curr_Y.cuda()
			curr_X = curr_X.cuda()

			curr_X = curr_X.float()  # FOR simple LSTM or LSTM+ATTENTION model
			# curr_X = curr_X.view(len(curr_X),-1)   # FOR MLP/SVM
			
			optimizer.zero_grad()
			prds = model.forward_rgcn(curr_X,curr_Y)
			loss2 = loss_func(prds,curr_Y)
			loss2.backward()
			optimizer.step()
			
			del(curr_X)

			loss = loss2.to('cpu')
			del(loss2)
			train_loss += loss.item()
			logits7 = prds.to('cpu')
			del(prds)
			results = (logits7.argmax(dim=1))
			labels = curr_Y.to('cpu')
			del(curr_Y)
			torch.cuda.empty_cache()

			#For overall accuracies
			for h in range(len(results)):
				if(results[h].item()==labels[h].item()):
					train_crct += 1
					train_cntr += 1
				else:
					train_cntr += 1

	        #For class accuracies
			for h in range(len(results)):
			#print(results[h],labels[h],nodefts[h])
				if(results[h].item()==labels[h].item()):
					train_class_crcts[labels[h].item()] += 1
					train_class_cnts[labels[h].item()] += 1
				else:
					train_class_cnts[labels[h].item()] += 1

		torch.cuda.empty_cache()    
		train_acc = train_crct / train_cntr

		#VALIDATION SET TESTING
		model.eval()

		batches = X_test.shape[0]/batch_size_val
		if(not(X_test.shape[0]%batch_size_val==0)):
			batches += 1
		
		batches = int(batches)
		for j in range(batches):
			curr_X = X_test[j*batch_size_val:min(X_test.shape[0],(j+1)*batch_size_val),:]
			curr_Y = Y_test[j*batch_size_val:min(X_test.shape[0],(j+1)*batch_size_val)]

			curr_Y = curr_Y.cuda()
			curr_X = curr_X.cuda()

			curr_X = curr_X.float()
			# curr_X = curr_X.view(len(curr_X),-1)

			prds = model.forward_rgcn(curr_X,curr_Y)
			loss2 = loss_func(prds,curr_Y)
			del(curr_X)

			loss = loss2.to('cpu')
			del(loss2)
			val_loss += loss.item()
			logits7 = prds.to('cpu')
			del(prds)
			results = (logits7.argmax(dim=1))
			labels = curr_Y.to('cpu')
			del(curr_Y)
			torch.cuda.empty_cache()

			for h in range(len(results)):
				if(results[h].item()==labels[h].item()):
					val_crct += 1
					val_cntr += 1
				else:
					val_cntr += 1

			for h in range(len(results)):
				if(results[h].item()==labels[h].item()):
					val_class_crcts[labels[h].item()] += 1
					val_class_cnts[labels[h].item()] += 1
				else:
					val_class_cnts[labels[h].item()] += 1

		scheduler.step(val_loss_schdlr)
		print('batches size',batch_size_train,batch_size_val)

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

		print("cl_ac",train_cl_ac,val_cl_ac)
		print(train_class_cnts,val_class_cnts)
		end=time.time()
		print("epoch time=",end-st)