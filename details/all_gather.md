# All-Gather Primitives

## Architecture & Workflow

```mermaid
flowchart TD
    S0[Shard on Rank 0] --> Gather[All-Gather]
    S1[Shard on Rank 1] --> Gather
    Gather --> Full[Full concatenated array on all Ranks]
```

## Overview

All-Gather collects sharded data blocks from all ranks and concatenates them, distributing the complete concatenated array back to every rank. It is the inverse of the Scatter operation.
