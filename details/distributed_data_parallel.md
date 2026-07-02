# Distributed Data Parallel (DDP)

## Architecture & Workflow

```mermaid
flowchart TD
    subgraph Node1 [Process 0]
        GPU0[GPU 0]
    end
    subgraph Node2 [Process 1]
        GPU1[GPU 1]
    end
    GPU0 <-->|NCCL All-Reduce| GPU1
```

## Overview

Distributed Data Parallel (DDP) runs a separate process per GPU. Gradient synchronization is performed using optimized NCCL All-Reduce, eliminating Python GIL bottlenecks and scaling efficiently across multiple machines.
