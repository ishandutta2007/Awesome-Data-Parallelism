# All-Reduce Primitives

## Architecture & Workflow

```mermaid
flowchart LR
    Input[Data on all ranks] --> Sum[Element-wise Sum]
    Sum --> Output[Global Sum on all ranks]
```

## Overview

All-Reduce is a collective operation that sums (or aggregates) values across all nodes and redistributes the global result back to all of them. It is the core primitive for gradient synchronization in data-parallel training.
