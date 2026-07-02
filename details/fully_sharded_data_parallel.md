# Fully Sharded Data Parallel (FSDP / ZeRO-Stage 3)

## Architecture & Workflow

```mermaid
flowchart TD
    Step1[Forward Pass: All-Gather parameters dynamically] --> Step2[Compute Forward]
    Step2 --> Step3[Evict non-sharded parameters]
    Step3 --> Step4[Backward Pass: All-Gather parameters & Reduce-Scatter gradients]
```

## Overview

FSDP shards all model parameters, gradients, and optimizer states across processes. Parameters are dynamically reconstructed via All-Gather before each forward/backward layer computation and immediately freed afterward, maximizing memory efficiency.
