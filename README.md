# Machine Learning Projects Collection

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

Welcome to my Machine Learning Projects Collection! This repository serves as a comprehensive resource for understanding and implementing various machine learning algorithms and architectures, from fundamental concepts to cutting-edge models.

## 🚀 Projects

### [GPT-2](./GPT-2/)
A complete implementation of OpenAI's GPT-2 model with pre-training and fine-tuning capabilities.
- **Features**: Multi-head attention, transformer architecture, BPE tokenization
- **Use Cases**: Text generation, language modeling, transfer learning
- **Tech Stack**: PyTorch, Transformers, Hugging Face Datasets

### [NanoGPT](./NanoGPT/)
A minimal but complete implementation of the GPT architecture, designed for educational purposes.
- **Features**: Pure PyTorch, efficient training, text generation
- **Use Cases**: Learning transformer architectures, small-scale language models
- **Tech Stack**: PyTorch, tqdm, numpy

### [Neural Network From Scratch](./Neural_Network_From_Scratch/)
Building neural networks from first principles using only NumPy.
- **Features**: Backpropagation, various layers, activation functions
- **Use Cases**: Educational, understanding deep learning fundamentals
- **Tech Stack**: NumPy, Matplotlib

### [makemore](./makemore/)
Character-level language models with multiple architectures.
- **Features**: Bigram, MLP, RNN, LSTM, GRU, and Transformer models
- **Use Cases**: Text generation, language modeling experiments
- **Tech Stack**: PyTorch, tqdm, matplotlib

### [Monte Carlo Tree Search](./Monte_Carlo_Tree_Search/)
Implementation of the Monte Carlo Tree Search algorithm for game AI.
- **Features**: Game-agnostic design, visualization, parallel simulations
- **Use Cases**: Game AI, decision making, planning
- **Tech Stack**: Python, NumPy, Matplotlib

## 🛠️ Getting Started

### Prerequisites

- Python 3.8+
- pip (Python package manager)
- Git (for cloning the repository)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ML_Stuff.git
   cd ML_Stuff
   ```

2. **Set up a virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. **Install project-specific dependencies**
   Each project has its own requirements. For example, to install requirements for GPT-2:
   ```bash
   cd GPT-2
   pip install -r requirements.txt
   ```

## 🎯 Usage

Each project contains detailed documentation and examples in its respective directory. Here's a quick start for running any project:

1. Navigate to the project directory
2. Follow the instructions in the project's README.md
3. Run the main script or Jupyter notebook

Example for running the NanoGPT training:
```bash
cd NanoGPT
python train.py
```

## 📊 Project Structure

```
ML_Stuff/
├── GPT-2/                  # GPT-2 implementation
│   ├── pre-training/       # Pre-training scripts
│   ├── notebooks/          # Jupyter notebooks for exploration
│   └── README.md           # Project documentation
│
├── NanoGPT/                # Minimal GPT implementation
│   ├── train.py            # Training script
│   ├── model.py            # Model architecture
│   └── README.md           # Documentation
│
├── Neural_Network_From_Scratch/  # NumPy neural networks
│   ├── neural_network.py   # Core implementation
│   └── README.md           # Documentation
│
├── makemore/               # Character-level language models
│   ├── makemore.py         # Main implementation
│   └── README.md           # Documentation
│
├── Monte_Carlo_Tree_Search/ # MCTS implementation
│   ├── mcts.py             # Core algorithm
│   └── README.md           # Documentation
│
└── README.md               # This file
```

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Report bugs**: Open an issue with detailed reproduction steps
2. **Suggest features**: Share your ideas for improvements
3. **Submit code**: Fork the repo and open a pull request

Please read our [Contributing Guidelines](CONTRIBUTING.md) for more details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📚 Resources

- [PyTorch Documentation](https://pytorch.org/docs/stable/index.html)
- [Deep Learning Book](https://www.deeplearningbook.org/)
- [CS231n: Convolutional Neural Networks](http://cs231n.stanford.edu/)
- [The Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html)

## 🙏 Acknowledgments

- The open-source community for their invaluable contributions
- All the researchers and developers who have shared their knowledge
- Special thanks to the creators of PyTorch, NumPy, and other essential libraries
