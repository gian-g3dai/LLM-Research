# LLM-Research

A research repository for LLM (Large Language Model) fine-tuning, inference, and experimentation — with a focus on Code Llama models and distributed training via PyTorch FSDP.

## Overview

This project provides tools and scripts for:

- **Fine-tuning Code Llama** models for domain-specific tasks
- **Distributed training** using PyTorch Fully Sharded Data Parallel (FSDP)
- **CUDA graph optimization** for accelerated training and inference
- **PDF data mining** for preprocessing training corpora
- **Cross-entropy loss splitting** for memory-efficient training
- **Recipe-based experiment configurations** for reproducible research

## Repository Structure

```
LLM-Research/
├── recipes/              # Experiment configuration recipes (set 1)
├── recipes_2/            # Experiment configuration recipes (set 2)
├── scripts/              # Utility scripts (e.g., checkpoint conversion)
├── tests/                # Unit and integration tests
├── unakin/               # Core library / package modules
├── .vscode/              # VS Code workspace settings
├── main.py               # Primary entry point
├── fsdp_test.py          # FSDP distributed training tests
├── fsdp_cuda_graph_test.py  # FSDP + CUDA graph integration tests
├── mine_pdf.py           # PDF data extraction and preprocessing
├── split_ce.py           # Split cross-entropy loss implementation
├── tt.py                 # Utility / test script
├── setup.py              # Package installation configuration
├── requirements.txt      # Python dependencies
└── README.md
```

## Getting Started

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU(s)
- Code Llama model weights

### Environment Setup

1. **Create and activate the environment:**

   ```bash
   # Unzip the environment archive
   unzip ncs.zip -d /ncs

   # Activate the environment
   source /ncs/bin/activate
   ```

2. **Install the package in development mode:**

   ```bash
   python setup.py develop
   ```

3. **Download Code Llama models:**

   Place the Code Llama model weights into:

   ```
   $HOME/fsx/codellama/
   ```

4. **Convert the model checkpoint:**

   Convert the Llama checkpoint into a mono (single-file) checkpoint format:

   ```bash
   python scripts/convert_llama_checkpoint.py
   ```

   > See the script for example commands and usage details.

5. **Prepare the dataset:**

   Extract the preprocessed data from `dataset_unity.7z`.

6. **Run fine-tuning and inference:**

   ```bash
   python main.py
   ```

## Key Components

### Distributed Training (FSDP)

- `fsdp_test.py` — Validates FSDP distributed training across multiple GPUs
- `fsdp_cuda_graph_test.py` — Tests FSDP integration with CUDA graphs for optimized throughput

### Data Processing

- `mine_pdf.py` — Extracts and preprocesses text from PDF documents for training data pipelines
- `split_ce.py` — Implements split cross-entropy loss for memory-efficient training on long sequences

### Experiment Recipes

The `recipes/` and `recipes_2/` directories contain configuration files for reproducible experiments with varying hyperparameters, model sizes, and training strategies.
