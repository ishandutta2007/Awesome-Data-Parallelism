# Zero Redundancy & Fully Sharded Parameter Era

## Architecture & Workflow

```mermaid
flowchart TD
    subgraph GPU0 [GPU 0]
        P0[Parameters Part 1]
        G0[Gradients Part 1]
        O0[Optimizer States Part 1]
    end
    subgraph GPU1 [GPU 1]
        P1[Parameters Part 2]
        G1[Gradients Part 2]
        O1[Optimizer States Part 2]
    end
```

## Overview

ZeRO (Zero Redundancy Optimizer) eliminates memory redundancy in data-parallel training by sharding the optimizer states, gradients, and model parameters across GPUs instead of replicating them. This unlocks the ability to train massive foundational models.
