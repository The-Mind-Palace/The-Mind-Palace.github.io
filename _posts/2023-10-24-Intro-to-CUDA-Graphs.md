---
layout: post
title: Intro to CUDA Graphs
categories: [cuda-graphs, inference-optimisations]
---

With advancements in GPU hardware and software, CPU overhead in launching CUDA kernels is beginning to become a bottleneck. This is especially observed in many HPC applications as deep neural network training and scientific simulations. Workloads using Python instead of C++ also experience this bottleneck [compared to C++, Python is 5x slower]. 

And while there are a few solutions to this bottleneck, this post is about CUDA Graphs.


## What is this? 




