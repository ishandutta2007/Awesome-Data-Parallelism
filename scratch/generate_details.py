import os

details_dir = r"C:\Users\ishan\Documents\Projects\Awesome-Data-Parallelism\details"
os.makedirs(details_dir, exist_ok=True)

pages = [
    {
        "filename": "parameter_server.md",
        "title": "Asynchronous Parameter Server Era",
        "diagram": """flowchart TD
    W1[Worker 1] -->|Push Gradients| PS[Parameter Server]
    W2[Worker 2] -->|Push Gradients| PS
    W3[Worker 3] -->|Push Gradients| PS
    PS -->|Pull Weights| W1
    PS -->|Pull Weights| W2
    PS -->|Pull Weights| W3""",
        "content": "The Parameter Server architecture relies on a centralized manager-worker layout. Worker nodes compute local gradients asynchronously and push them to the parameter server, which updates weights and broadcasts them back. The central node can become a major network bandwidth bottleneck in large clusters."
    },
    {
        "filename": "ring_allreduce.md",
        "title": "Synchronous Ring All-Reduce Era",
        "diagram": """flowchart LR
    GPU0[GPU 0] --> GPU1[GPU 1]
    GPU1 --> GPU2[GPU 2]
    GPU2 --> GPU3[GPU 3]
    GPU3 --> GPU0""",
        "content": "Ring All-Reduce arranges nodes in a logical ring topology. Each GPU sends data to its successor and receives data from its predecessor. Communication scales optimally with the number of devices and is independent of cluster size, eliminating the central server bottleneck."
    },
    {
        "filename": "zero_redundancy.md",
        "title": "Zero Redundancy & Fully Sharded Parameter Era",
        "diagram": """flowchart TD
    subgraph GPU0 [GPU 0]
        P0[Parameters Part 1]
        G0[Gradients Part 1]
        O0[Optimizer States Part 1]
    end
    subgraph GPU1 [GPU 1]
        P1[Parameters Part 2]
        G1[Gradients Part 2]
        O1[Optimizer States Part 2]
    end""",
        "content": "ZeRO (Zero Redundancy Optimizer) eliminates memory redundancy in data-parallel training by sharding the optimizer states, gradients, and model parameters across GPUs instead of replicating them. This unlocks the ability to train massive foundational models."
    },
    {
        "filename": "data_parallel.md",
        "title": "Data Parallel (DP / PyTorch Native Baseline)",
        "diagram": """flowchart TD
    Master[Master GPU] -->|Scatter Data & Replicate Weights| Worker1[GPU 1]
    Master -->|Scatter Data & Replicate Weights| Worker2[GPU 2]
    Worker1 -->|Gather Gradients| Master
    Worker2 -->|Gather Gradients| Master""",
        "content": "PyTorch DataParallel (DP) operates on a single machine by replicating the model on each GPU and using multi-threading to parallelize execution. However, it suffers from severe GIL (Global Interpreter Lock) bottlenecks and overheads."
    },
    {
        "filename": "distributed_data_parallel.md",
        "title": "Distributed Data Parallel (DDP)",
        "diagram": """flowchart TD
    subgraph Node1 [Process 0]
        GPU0[GPU 0]
    end
    subgraph Node2 [Process 1]
        GPU1[GPU 1]
    end
    GPU0 <-->|NCCL All-Reduce| GPU1""",
        "content": "Distributed Data Parallel (DDP) runs a separate process per GPU. Gradient synchronization is performed using optimized NCCL All-Reduce, eliminating Python GIL bottlenecks and scaling efficiently across multiple machines."
    },
    {
        "filename": "fully_sharded_data_parallel.md",
        "title": "Fully Sharded Data Parallel (FSDP / ZeRO-Stage 3)",
        "diagram": """flowchart TD
    Step1[Forward Pass: All-Gather parameters dynamically] --> Step2[Compute Forward]
    Step2 --> Step3[Evict non-sharded parameters]
    Step3 --> Step4[Backward Pass: All-Gather parameters & Reduce-Scatter gradients]""",
        "content": "FSDP shards all model parameters, gradients, and optimizer states across processes. Parameters are dynamically reconstructed via All-Gather before each forward/backward layer computation and immediately freed afterward, maximizing memory efficiency."
    },
    {
        "filename": "all_reduce.md",
        "title": "All-Reduce Primitives",
        "diagram": """flowchart LR
    Input[Data on all ranks] --> Sum[Element-wise Sum]
    Sum --> Output[Global Sum on all ranks]""",
        "content": "All-Reduce is a collective operation that sums (or aggregates) values across all nodes and redistributes the global result back to all of them. It is the core primitive for gradient synchronization in data-parallel training."
    },
    {
        "filename": "reduce_scatter.md",
        "title": "Reduce-Scatter Primitives",
        "diagram": """flowchart TD
    In[Data on all ranks] --> Red[Sum values across ranks]
    Red --> Scat[Scatter sharded results to individual ranks]""",
        "content": "Reduce-Scatter is a collective primitive that performs a reduction operation on vectors across ranks and scatters the reduced blocks evenly across them, so each rank receives a unique shard of the final result."
    },
    {
        "filename": "all_gather.md",
        "title": "All-Gather Primitives",
        "diagram": """flowchart TD
    S0[Shard on Rank 0] --> Gather[All-Gather]
    S1[Shard on Rank 1] --> Gather
    Gather --> Full[Full concatenated array on all Ranks]""",
        "content": "All-Gather collects sharded data blocks from all ranks and concatenates them, distributing the complete concatenated array back to every rank. It is the inverse of the Scatter operation."
    },
    {
        "filename": "network_overhang.md",
        "title": "The Network Communication Overhang and Interconnect Bottleneck",
        "diagram": """flowchart LR
    Backward[Backward computation of Layer L] -->|Overlap| Comm[All-Reduce of Layer L+1]
    Comm --> Finish[Synchronized step]""",
        "content": "In distributed scaling, slow network fabrics block high-performance GPUs. Mitigation techniques include gradient bucketing and overlapping communication (All-Reduce/Reduce-Scatter) in the background with current layer backward passes."
    },
    {
        "filename": "straggler_lock.md",
        "title": "The Straggler GPU Synchronization Lock",
        "diagram": """flowchart TD
    G1[Fast GPU 1] --> Wait[Barrier Wait]
    G2[Slow GPU 2 - Straggler] --> Wait
    Wait --> Sync[Synchronize & Update]""",
        "content": "Synchronous training progresses as fast as the slowest GPU (straggler). Dynamic monitoring frameworks, failovers, thermal management, and robust health checks are required to resolve and prevent cluster-wide stalls."
    },
    {
        "filename": "llm_pretraining.md",
        "title": "Pre-Training Multi-Trillion Token Foundation LLMs (Megatron-LM / DeepSpeed Clusters)",
        "diagram": """flowchart TD
    subgraph 3D Parallelism
        DP[Data Parallelism]
        TP[Tensor Parallelism]
        PP[Pipeline Parallelism]
    end""",
        "content": "Pre-training LLMs at scale uses hybrid 3D Parallelism, combining Data Parallelism (ZeRO/FSDP), Tensor Parallelism (sharding layers), and Pipeline Parallelism (sharding stages) to distribute the massive model and dataset."
    },
    {
        "filename": "video_diffusion.md",
        "title": "High-Volume Generative Video Diffusion Simulation Scaling (Sora Class)",
        "diagram": """flowchart LR
    Video[Spatio-temporal Video Cubes] --> Shards[Data Parallel Shards]
    Shards --> GPUs[Distributed GPU Array]""",
        "content": "Generative video diffusion models process large spatio-temporal video sequences. Sharding these massive video tokens across data-parallel systems allows processing diverse visual scenarios concurrently."
    },
    {
        "filename": "multimodal_alignment.md",
        "title": "Web-Scale Multimodal Representation Alignment Sprints (CLIP / OpenCLIP)",
        "diagram": """flowchart TD
    Images[Images Batch] --> ImageEnc[Image Encoder]
    Texts[Texts Batch] --> TextEnc[Text Encoder]
    ImageEnc & TextEnc --> Contrastive[Contrastive Matrix computation]""",
        "content": "Web-scale contrastive training (e.g., CLIP) aligns image and text embeddings. Using high-throughput Distributed Data Parallel (DDP) enables extremely large batch sizes to maximize negative samples and stabilize contrastive learning."
    }
]

for p in pages:
    filepath = os.path.join(details_dir, p["filename"])
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(f"# {p['title']}\n\n")
        f.write("## Architecture & Workflow\n\n")
        f.write("```mermaid\n" + p["diagram"] + "\n```\n\n")
        f.write("## Overview\n\n")
        f.write(p["content"] + "\n")

print("Generated 14 markdown pages successfully.")
