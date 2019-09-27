# Temporal MR-GCN
```
Spatio Temporal Modelling of Scenes through Multi Relational GCN:
Towards Accurate Vehicle Behaviour Classification." 
```
---------------------
### Base-line Implementation details(SVM based)
-------------
We propose an SVM based baseline with relational features and compare our method with it. We give the 3D locations of objects obtained from the Bird's eye View(mentioned in the paper) as a direct input. For each object in the video, we create a feature vector consisting of its distance and angle with all other nodes for T time stamps. The distance is a simple Euclidean distance. To account for the feature of object(lane-markings/Vehicles), we create a 2 dimensional one-hot vector{1,0} representing Vehicles and {0,1} representing static objects.
To create a feature vector for i-th object, we find distances and angles with all other nodes for T timestamps in the scene and concatenate them. 

Hence, The dimension for every object in the scene would include 2 dimensions from distance and angle with every other node for T time-stamps, hence (n-1) * 2 * T, and 2 dimensional vector input for encoding feature(lane/Vehicle). Hence , the total dimension for each node becomes:<br />
(n-1) * (2T+2).<br />
Here, as n changes with every graph,we pad zeros at the end to maintain a constant length for feature vector.
Algorithm for feature vector creation:<br/>

![SVM_algo](https://drive.google.com/uc?export=view&id=1TTSC9qqZeeNCiqz2un7JYfwqAAHS-_RN)

We combine both lane-changes into a single class and compare our method with SVM.(on APOLLO SCAPE dataset)<br />


| Method  | SVM |	Ours |
| ------------- | ------------- | ------------ |
| Moving away  | 52.1  |	85.3 |
| moving towards us  | 54.3  |	89.5 |
| Parked  | 54.3  |	94.8 |
| lane-change(L->R)  | -  |	84.1 |
| lane-change(R->L)  | -  |	86.4 |
| lane-change(overall)  | 52.8  | 84.8 |
| Overtake  | 57.2  | 72.3	 |

The training of SVM is done on a total of 10837(vehicles only) nodes with 30% validation. The dimension of each feature vector is 3960.

## Installation
##### requirements
```
dgl
pytorch == 1.2.0
pandas
numpy
matplotlib
tqdm
```
dgl can be installed from the official website, [dgl](https://www.dgl.ai/pages/start.html) based on cuda support.<br />
```
Installing without GPU:
pip install dgl
```
## Testing on apollo dataset 
```
git clone https://github.com/ma8sa/temporal-MR-GCN.git
cd temporal-MR-GCN
python3 lstm_rgcn_test_apollo.py
```
Graphs for all datasets can be downloaded from [graphs](https://drive.google.com/drive/folders/120UPpzhW0mgZUjKq30BskSdZHAg4Yt-Z?usp=sharing).<br />
**NOTE** : Make sure to extract the corresponding graphs and place it in the same folder where you are running the code from.

For information on how graphs are stored in npz files, go through the README file in the same link.
## Testing on Indian/Kitti dataset 
```
for indian,
python3 lstm_rgcn_test_ind_kitti.py indian
for kitti,
python3 lstm_rgcn_test_ind_kitti.py kitti
```
**NOTE** : Make sure to extract the corresponding graphs and place it in the same folder where you are running the code from.

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
| class accuracy(train)| 85 | 89 | 94 | 84 | 86 | 72 |  
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

### Dataset
-----------
We selected 3 main datasets to perform the experiments.
1. [Apollo](http://apolloscape.auto/scene.html) 
2. [Kitti](http://www.cvlibs.net/datasets/kitti/eval_tracking.php)
3. Indian

On apollo we have selected sequences from scene-parsing dataset and picked around 70 small sequences(each containing aroud 100 images) manually that include behaviours of our interest. Similarly on Kitti, we use tracking sequences 4,5,10 which are in line with our class requirement.
