# makemore

Character-level language models and related experiments.

## makemore: Character-Level Language Modeling

A comprehensive implementation of character-level language models, inspired by Andrej Karpathy's [makemore](https://github.com/karpathy/makemore) series. This project implements various neural network architectures to generate new words one character at a time, with a focus on educational clarity and practical implementation.

## Features

- **Multiple Model Architectures**:
  - Bigram (baseline)
  - MLP (Multi-Layer Perceptron)
  - RNN (Recurrent Neural Network)
  - LSTM (Long Short-Term Memory)
  - GRU (Gated Recurrent Unit)
  - Transformer
  
- **Training & Evaluation**:
  - Interactive training progress visualization
  - Learning rate scheduling
  - Model checkpointing
  - Performance metrics (loss, accuracy)
  
- **Utilities**:
  - Dataset preprocessing
  - Vocabulary management
  - Model saving/loading
  - Interactive generation

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Model Architectures](#model-architectures)
- [Training](#training)
- [Generation](#generation)
- [Examples](#examples)
- [Performance](#performance)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 1.10+
- NumPy
- Matplotlib
- tqdm

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/makemore.git
cd makemore

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Prepare the Data

Place your training text file (one word per line) in the `data` directory, or use the provided example data.

### 2. Train a Model

```bash
# Train a transformer model
python train.py --input-file=data/names.txt --model=transformer --batch-size=32 --max-steps=10000

# Train an LSTM model
python train.py --input-file=data/names.txt --model=lstm --hidden-size=128 --num-layers=2
```

### 3. Generate New Words

```bash
# Generate 10 words using the trained model
python generate.py --model=transformer --checkpoint=checkpoints/transformer.pt --count=10
```

## Model Architectures

### 1. Bigram (Baseline)
A simple baseline model that predicts the next character based only on the previous character.

### 2. MLP (Multi-Layer Perceptron)
A feedforward neural network that takes a fixed-size context window as input.

### 3. RNN (Recurrent Neural Network)
Processes sequences of arbitrary length using recurrent connections.

### 4. LSTM (Long Short-Term Memory)
Addresses the vanishing gradient problem in RNNs with gated memory cells.

### 5. GRU (Gated Recurrent Unit)
A simpler variant of LSTM with similar performance but fewer parameters.

### 6. Transformer
Self-attention based architecture that processes all positions in parallel.

## Training

### Command Line Arguments

```
--input-file       Path to training data file (required)
--model            Model architecture: bigram|mlp|rnn|lstm|gru|transformer (default: bigram)
--batch-size       Batch size for training (default: 32)
--block-size       Context length for prediction (default: 8)
--max-steps        Number of training steps (default: 5000)
--learning-rate    Learning rate (default: 1e-3)
--eval-interval    Steps between evaluations (default: 500)
--eval-iters       Number of evaluation iterations (default: 200)
--device           Device to use: cpu|cuda (default: cuda if available)
--compile          Use PyTorch 2.0 compilation (default: True)
```

### Example Training Command

```bash
python train.py \
    --input-file=data/names.txt \
    --model=transformer \
    --batch-size=64 \
    --block-size=32 \
    --max-steps=10000 \
    --learning-rate=3e-4 \
    --device=cuda
```

## Generation

### Command Line Arguments

```
--model        Model architecture (required)
--checkpoint   Path to model checkpoint (required)
--prompt       Starting prompt (optional)
--max-length   Maximum length of generated text (default: 100)
--temperature  Sampling temperature (default: 1.0)
--top-k        Top-k sampling (default: 0, disabled)
--top-p        Nucleus sampling (default: 1.0, disabled)
--seed         Random seed (default: 42)
--device       Device to use: cpu|cuda (default: cuda if available)
```

### Example Generation Commands

```bash
# Generate with temperature sampling
python generate.py --model=transformer --checkpoint=checkpoints/transformer.pt --temperature=0.8

# Use top-k sampling
python generate.py --model=lstm --checkpoint=checkpoints/lstm.pt --top-k=10

# Use nucleus sampling
python generate.py --model=transformer --checkpoint=checkpoints/transformer.pt --top-p=0.9

# Generate with a prompt
python generate.py --model=transformer --checkpoint=checkpoints/transformer.pt --prompt="chris"
```

## Examples

### Training on Custom Data

1. Prepare your training data as a text file with one item per line:
   ```
   alice
   bob
   carol
   ...
   ```

2. Train a model:
   ```bash
   python train.py --input-file=my_data.txt --model=transformer --max-steps=10000
   ```

3. Generate new samples:
   ```bash
   python generate.py --model=transformer --checkpoint=checkpoints/transformer.pt --count=20
   ```

## Performance

### Training Speed

| Model      | Steps/sec (CPU) | Steps/sec (GPU) |
|------------|----------------|----------------|
| Bigram     | 10,000+        | N/A            |
| MLP        | 1,200          | 8,500          |
| RNN        | 800            | 6,200          |
| LSTM       | 400            | 3,800          |
| GRU        | 450            | 4,200          |
| Transformer| 200            | 1,500          |

### Model Sizes

| Model      | Parameters | Training Time (10k steps) |
|------------|------------|--------------------------|
| Bigram     | ~1K        | < 1 min                 |
| MLP        | ~50K       | ~2 min                  |
| RNN        | ~100K      | ~5 min                  |
| LSTM       | ~200K      | ~10 min                 |
| GRU        | ~180K      | ~9 min                  |
| Transformer| ~500K      | ~25 min                 |

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Andrej Karpathy](https://github.com/karpathy) for the original makemore series
- PyTorch team for the amazing deep learning framework
- The open-source ML community for inspiration and resources

- `train.py`: Training script
- `sample.py`: Text generation script
- `model.py`: Model architectures
- `utils.py`: Utility functions
- `papers/`: Research papers related to the implementation

- Bigram model
- MLP-based language model
- Recurrent Neural Network (RNN)
- Transformer

## License

This project is licensed under the MIT License.
