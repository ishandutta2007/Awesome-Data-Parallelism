# Asynchronous Parameter Server Era

## Architecture & Workflow

```mermaid
flowchart TD
    W1[Worker 1] -->|Push Gradients| PS[Parameter Server]
    W2[Worker 2] -->|Push Gradients| PS
    W3[Worker 3] -->|Push Gradients| PS
    PS -->|Pull Weights| W1
    PS -->|Pull Weights| W2
    PS -->|Pull Weights| W3
```

## Overview

The Parameter Server architecture relies on a centralized manager-worker layout. Worker nodes compute local gradients asynchronously and push them to the parameter server, which updates weights and broadcasts them back. The central node can become a major network bandwidth bottleneck in large clusters.
