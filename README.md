## large-scale-OT-mapping-TF

Tensorflow Implementation of the following paper:
```
Title:	
Large-Scale Optimal Transport and Mapping Estimation
Authors:	
Seguy, Vivien; Bhushan Damodaran, Bharath; Flamary, Rémi; Courty, Nicolas; Rolet, Antoine; Blondel, Mathieu
Publication:	
eprint arXiv:1711.02283
Publication Date:	
11/2017
Origin:	
ARXIV
Keywords:	
Statistics - Machine Learning
Comment:	
10 pages, 4 figures
Bibliographic Code:	
2017arXiv171102283S
```
[on arXiv](https://arxiv.org/abs/1711.02283)

### Some notes

This repository does not contain an implementation of the entire experiment of the paper. Instead,
it confirms the thesis's core algorithm in a small toy example.

Unlike paper, batch-wise optimization is not done.

L2-based regularization is not implemented. Entropy-based regularization only.

### Requirements
```
python3
tensorflow
matplotlib
seaborn
...
```

## Results

##### Source and Target
Source points are green and target points are red.

##### Monge Map Estimation
Source points are green and transported points are blue.

##### KDE on transported distribution

#### Author
@mikigom (Junghoon Seo)

sjh@satreci.com