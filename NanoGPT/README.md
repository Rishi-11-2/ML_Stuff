# NanoGPT: Minimal GPT Implementation

A minimal but complete implementation of the GPT (Generative Pre-trained Transformer) model in PyTorch. This implementation is designed for educational purposes, providing a clear and concise codebase that demonstrates the core concepts of transformer-based language models.

## Features

- **Complete Implementation**: Includes all key components of GPT
- **Minimal Dependencies**: Only requires PyTorch and standard libraries
- **Educational Focus**: Clean, well-commented code with explanations
- **Interactive Notebooks**: Jupyter notebooks for visualization and experimentation
- **Efficient Training**: Implements gradient checkpointing and mixed precision

## Model Architecture

NanoGPT implements the core components of the GPT architecture:

- **Transformer Decoder**: Stack of identical transformer layers
- **Multi-Head Self-Attention**: With causal masking for autoregressive generation
- **Position-wise Feed-Forward Networks**: Two-layer MLP with GELU activation
- **Layer Normalization**: Applied before attention and feed-forward layers
- **Residual Connections**: Around each sub-layer
- **Learnable Position Embeddings**: For sequence position information

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- tqdm (for progress bars)
- matplotlib (for visualization)

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/nanogpt.git
   cd nanogpt
   ```

2. Install dependencies:
   ```bash
   pip install torch tqdm matplotlib
   ```

### Quick Start

1. **Train the model** (on CPU/GPU):
   ```bash
   python train.py \
       --batch_size 64 \
       --block_size 256 \
       --n_embd 384 \
       --n_head 6 \
       --n_layer 6 \
       --max_iters 5000 \
       --eval_interval 500
   ```

2. **Generate text** with the trained model:
   ```bash
   python sample.py --prompt "The meaning of life is"
   ```

## Project Structure

- `train.py`: Main training script
- `sample.py`: Text generation script
- `model.py`: Model architecture implementation
- `utils.py`: Utility functions
- `bigram.py`: Simple bigram language model baseline
- `trigram.py`: Trigram language model implementation
- `input.txt`: Sample training text
- `gpt_dev.ipynb`: Development notebook with experiments

## Training Options

### Model Architecture
- `--n_layer`: Number of transformer layers (default: 6)
- `--n_head`: Number of attention heads (default: 6)
- `--n_embd`: Embedding dimension (default: 384)
- `--block_size`: Context length (default: 256)
- `--dropout`: Dropout rate (default: 0.2)

### Training
- `--batch_size`: Batch size (default: 64)
- `--max_iters`: Maximum training iterations (default: 10,000)
- `--learning_rate`: Learning rate (default: 3e-4)
- `--eval_interval`: Evaluation interval (default: 500)
- `--eval_iters`: Number of evaluation iterations (default: 200)
- `--gradient_accumulation_steps`: Gradient accumulation steps (default: 1)

### System
- `--device`: Device to use (default: cuda if available)
- `--dtype`: Data type (default: bfloat16 if available, else float16)
- `--compile`: Use PyTorch 2.0 compilation (default: True)

## Examples

### Training Command

Train a larger model with gradient accumulation:
```bash
python train.py \
    --n_layer 12 \
    --n_head 12 \
    --n_embd 768 \
    --batch_size 16 \
    --gradient_accumulation_steps 4 \
    --max_iters 20000
```

### Generation Examples

Generate text with temperature sampling:
```bash
python sample.py --temperature 0.8 --top_k 40 --max_new_tokens 500
```

## Educational Resources

### Key Concepts Demonstrated
1. **Autoregressive Language Modeling**
2. **Self-Attention Mechanism**
3. **Transformer Decoder Architecture**
4. **Efficient Training Techniques**
5. **Text Generation Strategies**

### Learning Path
1. Start with `bigram.py` to understand the basics
2. Move to `trigram.py` for n-gram modeling
3. Study `model.py` for the transformer implementation
4. Explore training in `train.py`
5. Experiment with generation in `sample.py`

## Performance

On an NVIDIA V100 GPU:
- Training: ~5,000 tokens/second
- Inference: ~100 tokens/second (with beam search)

## Contributing

Contributions are welcome! Please follow these guidelines:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [GPT-2 Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)
