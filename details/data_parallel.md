# Data Parallel (DP / PyTorch Native Baseline)

## Architecture & Workflow

```mermaid
flowchart TD
    Master[Master GPU] -->|Scatter Data & Replicate Weights| Worker1[GPU 1]
    Master -->|Scatter Data & Replicate Weights| Worker2[GPU 2]
    Worker1 -->|Gather Gradients| Master
    Worker2 -->|Gather Gradients| Master
```

## Overview

PyTorch DataParallel (DP) operates on a single machine by replicating the model on each GPU and using multi-threading to parallelize execution. However, it suffers from severe GIL (Global Interpreter Lock) bottlenecks and overheads.
