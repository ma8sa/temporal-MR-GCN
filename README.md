# Temporal MR-GCN
```
Towards Accurate Vehicle Behaviour Classification
With Multi Relational Graph Convolutional Networks
```
For any queries mail : saisravan.m@research.iiit.ac.in (or)
mahtab.sandhu@gmail.com


<!---
---------------------
### Base-line Implementation details(SVM based)
-------------
We propose an SVM based baseline with relational features and compare our method with it. We give the 3D locations of objects obtained from the Bird's eye View(mentioned in the paper) as a direct input. For each object in the video, we create a feature vector consisting of its distance and angle with all other nodes for T time stamps. The distance is a simple Euclidean distance. To account for the feature of object(lane-markings/Vehicles), we create a 2 dimensional one-hot vector{1,0} representing Vehicles and {0,1} representing static objects.
To create a feature vector for i-th object, we find distances and angles with all other nodes for T timestamps in the scene and concatenate them. 
>
Hence, The dimension for every object in the scene would include 2 dimensions from distance and angle with every other node for T time-stamps, hence (n-1) * 2 * T, and 2 dimensional vector input for encoding feature(lane/Vehicle). Hence , the total dimension for each node becomes:<br />
(n-1) * (2T+2).<br />
Here, as n changes with every graph,we pad zeros at the end to maintain a constant length for feature vector.
Algorithm for feature vector creation:<br/>
>
![SVM_algo](https://drive.google.com/uc?export=view&id=1TTSC9qqZeeNCiqz2un7JYfwqAAHS-_RN)
>
We combine both lane-changes into a single class and compare our method with SVM.(on APOLLO SCAPE dataset)<br />

The training of SVM is done on a total of 10837(vehicles only) nodes with 30% validation. The dimension of each feature vector is 3960.
>
--->

---------------------
### Base-line Implementation details(SVM based)
-------------

Apart from comparisons in the paper, we use Structural-RNN, a LSTM based graph network. Since the tasks in their paper confine only to driver-anticipation, we use one of their methods similar to our task. Specifically, we use the **detection** method of **activity-anticipation** mentioned in the paper due to the similarity in the architecture and task . We use *Vehicles* as *Humans* and *Lane Markings* as *Objects* in their architecture for our purpose. Similar to the Human-Object, Human-Human and Object-Object interactions, we observe the Vehicle-Lane, Vehicle-Vehicle and Lane-Lane interactions for all time-steps. We give features similar to the baselines in the paper.

| Method  | St-RNN |	Ours |
| ------------- | ------------- | ------------ |
| Moving away  | 76  |	85.3 |
| moving towards us  | 51 |	89.5 |
| Parked  | 83  |	94.8 |
| lane-change(L->R)  | 52  |	84.1 |
| lane-change(R->L)  | 57  |	86.4 |
| Overtake  | 63  | 72.3	 |

<!-- | lane-change(overall)  | 52.8  | 84.8 |-->

### Dataset
-----------
We selected 3 main datasets to perform the experiments.
1. [Apollo](http://apolloscape.auto/scene.html) 
2. [Kitti](http://www.cvlibs.net/datasets/kitti/eval_tracking.php)
3. Indian


**Graphs for all datasets can be downloaded from [graphs](https://drive.google.com/drive/folders/120UPpzhW0mgZUjKq30BskSdZHAg4Yt-Z?usp=sharing).<br />
For information on how each graph is stored as a npz file, go through the README file in the same link.**


On apollo we have selected sequences from scene-parsing dataset and picked around 70 small sequences(each containing aroud 100 images) manually that include behaviours of our interest. Similarly on Kitti, we use tracking sequences 4,5,10 which are in line with our class requirement.

### Installation
--------------
##### Requirements
```
dgl
pytorch == 1.2.0
pandas
numpy
tqdm
```

#### Installing without GPU:
```
pip3 install requirements.txt
```
To install and use GPU for dgl, cuda support can be installed from their official website, [dgl](https://www.dgl.ai/pages/start.html) .<br /> 
And set *use_cuda = 1* in training/testing codes.

## Testing and Training on apollo dataset 
```
git clone https://github.com/ma8sa/temporal-MR-GCN.git
cd temporal-MR-GCN
python3 lstm_rgcn_test_apollo.py
```
**NOTE** : Make sure to extract the corresponding graphs(*lstm_graphs_apollo*) and place it in the same folder where you are running the code from.


## Testing on Indian/Kitti dataset (Transfer Learning)
```
for indian,
python3 lstm_rgcn_test_ind_kitti.py indian
for kitti,
python3 lstm_rgcn_test_ind_kitti.py kitti
```
**NOTE** : Make sure to extract the corresponding graphs (*lstm_graphs_kitti* for **kitti** and *lstm_graphs_indian* for **indian**) and place it in the same folder where you are running the code from.
### RESULTS
---------
0->Move forward<br />
1->Moving towards us<br />
2->Parked<br />
3-> lane-change(L->R)<br />
4-> lane-change(R->L)<br />
5-> Overtake

##### Results on Apollo
|  | 0 | 1 | 2 | 3 | 4 | 5 |
| ------------- | ------------- | ------------ | ------------ | ------------ | ------------ | ------------ |
| class accuracy(train)| 95 | 98 | 97 | 96 | 96 | 97 |  
| class counts(train)  | 2673 | 685 | 3574 |424  | 452 | 525  |
| class accuracy(val)  | 85 | 89 | 94 | 84 | 86 | 72 |
| class counts(val)  | 814 | 237 | 1415 | 162 | 130 | 73 |

Since the number of cars showing overtake behaviour are less, we augmented and added few synthetic-graphs and augmented data to train data only for apollo. With a little more data, the model can clearly learn overtake too above 80%, as number of synthetic graphs added were too low 76.

##### Results on Kitti tested with weights trained on Apollo
|  | 0 | 1 | 2 |
| ------------- | ------------- | ------------ | ------------ |
| class accuracy| 99 | 98 | 98 |
| class counts  | 504 | 230 | 674 |

##### Results on Indian tested with weights trained on Apollo
|  | 0 | 1 | 2 |
| ------------- | ------------- | ------------ | ------------ |
| class accuracy| 99 | 92 | 99 |
| class counts  | 324 | 229 | 2547 |



<!---
### Attention Explanantion
-----------
Due to space constraint in the paper, we have defined attention as a module in the paper. Here, we give it's working and explanation.<br/>
To weight the outputs from LSTM(which are ordered w.r.t time), we use attention as a weighted sum for predicting the output.<br/>
>
Given output from LSTM as L<sub>g</sub>,
we define a HEAD as triplet containing Query(Q),Key(K),Value(V). The query, Key and Values are learnable intermediate parameters. Q and K are used to find which values of input are similar/highly related and V is to weight them. Hence, the equation becomes : 
>
![attention_eqn](https://drive.google.com/uc?export=view&id=1AsejV-js_mxJ3oJnoLqMDZwBGRBrgj0B)
>
dk is the sacling factor(from paper). This is applied for all time-stamps.<br/> 
As dimension of L<sub>g</sub> is N x T x d<sub>2</sub>, attention using Q,K,V on **each node** gives, T x d<sub>3</sub> output. **Attention applies the above equation for all time-stamps, hence the T x d<sub>3</sub> output**.<br/>
If h heads are available, all heads are concatenated not across time but across d<sub>3</sub> dimension. Hence, output dimension remains same as T x d<sub>3</sub>, as we finally project to input dimension for output from attention.
![mh eqn](https://drive.google.com/uc?export=view&id=1RGs2zFIPcZA6t3jTy0S07BM-c_6rG3jQ)
>
where head<sub>i</sub> = Attention(Q,K<sub>i</sub>,V<sub>i</sub>).<br/>
The final out put of attention is T x d<sub>3</sub> for **each node**.
--->
