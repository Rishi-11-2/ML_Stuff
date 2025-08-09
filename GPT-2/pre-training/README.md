# GPT-2 Pre-training

This directory contains the core components for pre-training the GPT-2 model from scratch. The implementation is optimized for both single-GPU and multi-GPU training using PyTorch's DistributedDataParallel.

## Files

- `train_gpt2.py`: Main training script with training loop and model definition
- `fineweb.py`: Data loading utilities for the FineWeb dataset
- `hellaswag.py`: HellaSwag dataset loading and evaluation utilities
- `input.txt`: Sample text data for testing

## Model Architecture

The implementation includes:

- **Transformer Decoder**: 12-layer transformer decoder with masked self-attention
- **Attention**: Multi-head attention with 12 heads and 768-dimensional embeddings
- **Feed-Forward**: Position-wise feed-forward networks with GELU activation
- **Layer Normalization**: Applied before attention and feed-forward layers
- **Residual Connections**: Around each sub-layer
- **Tokenization**: Byte Pair Encoding (BPE) with a vocabulary size of 50,257

## Training Process

### Data Loading

1. **FineWeb Dataset**: Large-scale web text dataset
   - Automatically downloaded and processed
   - Streamed to avoid loading entire dataset into memory

2. **HellaSwag Benchmark**: Used for evaluation
   - Measures model's commonsense reasoning
   - Provides validation metrics during training

### Training Command

```bash
torchrun --nproc_per_node=8 train_gpt2.py \
    --batch_size 8 \
    --block_size 1024 \
    --n_layer 12 \
    --n_head 12 \
    --n_embd 768 \
    --learning_rate 6e-4 \
    --min_lr 6e-5 \
    --warmup_iters 2000 \
    --lr_decay_iters 600000 \
    --max_iters 600000 \
    --eval_interval 2000 \
    --eval_iters 200 \
    --gradient_accumulation_steps 5 \
    --compile True \
    --out_dir ./checkpoints
```

### Key Arguments

#### Model Architecture
- `--n_layer`: Number of transformer layers (default: 12)
- `--n_head`: Number of attention heads (default: 12)
- `--n_embd`: Embedding dimension (default: 768)
- `--block_size`: Context length (default: 1024)

#### Training
- `--batch_size`: Batch size per GPU (default: 8)
- `--gradient_accumulation_steps`: Accumulate gradients over multiple steps (default: 5)
- `--max_iters`: Maximum number of training iterations (default: 600,000)
- `--learning_rate`: Initial learning rate (default: 6e-4)
- `--min_lr`: Minimum learning rate (default: 6e-5)
- `--warmup_iters`: Learning rate warmup steps (default: 2,000)
- `--lr_decay_iters`: Learning rate decay steps (default: 600,000)

#### Evaluation
- `--eval_interval`: Evaluate model every N steps (default: 2,000)
- `--eval_iters`: Number of iterations per evaluation (default: 200)
- `--eval_only`: Run evaluation only (default: False)

#### System
- `--compile`: Use PyTorch 2.0 compilation (default: True)
- `--device`: Device to use (default: cuda if available)
- `--dtype`: Data type (default: bfloat16 if supported, else float16)
- `--out_dir`: Output directory for checkpoints (default: ./out)

## Checkpoints

Checkpoints are saved in the following format:
```
checkpoints/
  ├── iter-{iter}-loss-{loss:.4f}.pt  # Model checkpoints
  ├── best_val_loss.pt                # Best validation checkpoint
  └── config.json                     # Training configuration
```

## Performance Optimization

1. **Gradient Accumulation**: Allows for larger effective batch sizes
2. **Gradient Clipping**: Prevents exploding gradients
3. **Mixed Precision Training**: Uses FP16/BF16 for faster training
4. **Gradient Checkpointing**: Reduces memory usage
5. **PyTorch Compilation**: Optimizes model execution

## Evaluation

Model performance is evaluated using:
- Training loss (cross-entropy)
- Validation loss
- HellaSwag accuracy (zero-shot)

## Requirements

- Python 3.8+
- PyTorch 2.0+
- tqdm
- numpy
- transformers

## References

- [GPT-2 Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Hugging Face Transformers](https://huggingface.co/transformers/)
- [FineWeb Dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb)
