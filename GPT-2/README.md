# GPT-2 Implementation

This directory contains a complete implementation of OpenAI's GPT-2 model, including pre-training and fine-tuning capabilities. The implementation is built using PyTorch and follows the original architecture described in the [GPT-2 paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf).

## Directory Structure

- `pre-training/`: Core training scripts and utilities
  - `train_gpt2.py`: Main training script
  - `fineweb.py`: Data loading utilities for FineWeb dataset
  - `hellaswag.py`: HellaSwag dataset loading and evaluation
  - `input.txt`: Sample training data
- `notebooks/`: Jupyter notebooks for exploration and visualization

## Model Architecture

The implementation includes all key components of GPT-2:

- **Multi-head self-attention** with 12 attention heads (by default)
- **Layer normalization** before attention and feed-forward layers
- **Position-wise feed-forward networks** with GELU activation
- **Byte pair encoding (BPE)** tokenization
- **Causal attention masks** to prevent information leakage
- **Gradient checkpointing** for memory efficiency

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 1.10+
- Transformers library
- tqdm for progress bars
- Other dependencies in `requirements.txt`

### Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Training

To train the model:

```bash
cd pre-training
python train_gpt2.py \
    --batch_size 8 \
    --block_size 1024 \
    --n_layer 12 \
    --n_head 12 \
    --n_embd 768 \
    --learning_rate 6e-4 \
    --max_iters 10000 \
    --eval_interval 1000
```

### Key Arguments

- `--batch_size`: Batch size for training
- `--block_size`: Context length (max sequence length)
- `--n_layer`: Number of transformer layers
- `--n_head`: Number of attention heads
- `--n_embd`: Embedding dimension
- `--learning_rate`: Learning rate
- `--max_iters`: Maximum number of training iterations
- `--eval_interval`: Evaluate model every N iterations

## Fine-tuning

The model can be fine-tuned on custom datasets by modifying the data loading logic in `fineweb.py` or `hellaswag.py`.

## Evaluation

Model performance can be evaluated using the HellaSwag benchmark or custom evaluation scripts.

## License

This implementation is provided under the MIT License. See the LICENSE file for details.

## References

- [Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [FineWeb Dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb)
