---
layout: page
title: vAttention
description: Dynamic Memory Management for Serving LLMs without PagedAttention
img: assets/img/projects/vAttention_ss1.png
importance: 1
category: Systems for ML
---

Efficient management of GPU memory is essential for high throughput LLM inference. Prior systems used to reserve KV-cache memory ahead-of-time that resulted in wasted capacity due to internal fragmentation. Inspired by demand paging, vLLM proposed PagedAttention to enable dynamic memory allocation for KV-cache. This approach eliminates fragmentation and improves serving throughout. However, to be able to allocate physical memory dynamically, PagedAttention changes the layout of KV-cache from contiguous virtual memory to non-contiguous virtual memory. As a consequence, one needs to rewrite the attention kernels to support paging, and implement a memory manager in the serving framework. This results in both performance and programming overheads, as well as portability challenges in adopting state-of-the-art attention kernels. 

We solve these problems by storing the KV-cache in contiguous virtual memory and leverages OS support for on-demand allocation of physical memory.
<!-- We address this problem both in supervised settings, as well as unsupervised setting for pretraining large models on large datasets. -->

<div class="row">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.html path="assets/img/projects/vAttention_ss1.png" title="vAttention Allocation" class="img-fluid rounded z-depth-1" %}
    </div>
</div>

Checkout the following papers to know more:

<div class="publications">
{% bibliography -f {{ site.scholar.bibliography }} -q @*[vattention=true]* %}
</div>




