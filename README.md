# Eigenvector Grouping
This repository contains code implementation for the paper: "Eigenvector Grouping for Point Cloud Vessel Labeling" from the Geometric Deep Learning in Medical Imaging (GeoMedIA) 2022 Workshop.

Link to the paper: https://openreview.net/forum?id=leroKD5uuNf

# Introduction
In this work a novel point gruping technique called Eigenvector Grouping is introduced to tackle a problem of labeling vessels in voxel-based vessel segmentation. The tasks aims to filter out artefact vessels and correctly detect disconnected branches of vessels of interest. 

Grouping along eigenvectors of local point neighbourhoods is proposed to perform topology-aware grouping instead of previous topology-agnostic approaches e.g. ball-query, KNN.

# Installation
The code utilizes the Pointnet2/Pointnet++ PyTorch library by Erik Wijmans (https://github.com/erikwijmans/Pointnet2_PyTorch). The library is extended with implementations of `Geometry Aware Grouping (GAG)` (https://arxiv.org/abs/2012.07262) and our `Eigenvector Grouping (EVG)`.

To build CUDA kernels, navigate to `src` and run command below:  
```
pip install pointnet2_ops_lib/
````
