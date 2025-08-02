# ML Projects Collection

This repository contains multiple machine learning projects, including:

1. Handwritten Digit Recognition with Neural Networks
2. Monte Carlo Tree Search for Answer Generation

## 1. Handwritten Digit Recognition

This project implements a two-layer neural network from scratch using NumPy to classify handwritten digits from the MNIST dataset.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Code Structure](#code-structure)
- [License](#license)

## Overview

This implementation demonstrates a simple yet effective neural network for digit recognition using:
- Input layer: 784 neurons (28x28 pixel images)
- Hidden layer: 64 neurons with ReLU activation
- Output layer: 10 neurons (digits 0-9) with SoftMax activation

The network is trained using gradient descent with backpropagation.

## Features

- Pure NumPy implementation (no deep learning frameworks)
- Mini-batch gradient descent
- ReLU activation for hidden layer
- SoftMax activation for output layer
- Cross-entropy loss function
- Training progress monitoring
- Validation accuracy calculation

## Requirements

- Python 3.6+
- NumPy
- Pandas
- Matplotlib (for data visualization)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/ML_Stuff.git
   cd ML_Stuff
   ```

2. Install the required packages:
   ```bash
   pip install numpy pandas matplotlib
   ```

## Usage

1. Ensure you have the MNIST dataset in the correct location or modify the path in `init_data()` function.

2. Run the neural network:
   ```bash
   python Neural_Network.py
   ```

## Model Architecture

The network consists of:
- **Input Layer**: 784 neurons (28x28 pixel images)
- **Hidden Layer**: 10 neurons with ReLU activation
- **Output Layer**: 10 neurons (digits 0-9) with SoftMax activation

## Training

The model is trained using:
- Mini-batch gradient descent
- Learning rate: 0.01 (configurable)
- Number of iterations: 1000 (configurable)
- Training progress is printed every 50 iterations

## Evaluation

After training, the model's performance is evaluated on a separate validation set. The accuracy is printed after training completes.

## Results

Typical results include:
- Training accuracy: ~95-98%
- Validation accuracy: ~90-95%

## Code Structure

- `Neural_Network.py`: Main implementation file containing:
  - Data loading and preprocessing
  - Neural network initialization
  - Forward and backward propagation
  - Training loop
  - Evaluation functions

## 2. Monte Carlo Tree Search for Answer Generation

This project implements a Monte Carlo Tree Search (MCTS) algorithm for generating and refining answers to questions. The system uses a language model to generate, critique, and improve responses through an iterative process.

### Features

- **Tree-based Search**: Implements MCTS to explore different answer variations
- **Answer Generation**: Uses language models to generate potential answers
- **Self-Critique**: Automatically critiques and scores generated answers
- **Iterative Improvement**: Refines answers based on critique feedback
- **Multi-turn Reasoning**: Supports complex reasoning through multiple iterations

### How It Works

1. **Initialization**: Starts with seed answers or generates initial responses
2. **Selection**: Uses UCB1 algorithm to select the most promising node
3. **Expansion**: Generates new answer variations for promising paths
4. **Simulation**: Evaluates the quality of answers through self-critique
5. **Backpropagation**: Updates node statistics based on evaluation results

### Key Components

- `Node` class: Represents a node in the search tree with answer, visits, and value
- `MCTS` class: Implements the Monte Carlo Tree Search algorithm
- `rate_answer`: Evaluates answer quality using language model feedback
- `get_critique`: Generates detailed critiques of answers
- `improve_answer`: Refines answers based on critiques

### Example Usage

```python
question = "What is the capital of France?"
seed_answers = ["I don't know", "I'm not sure", "I can't say"]
mcts = MCTS(question, seed_answers, iterations=5)
best_answer = mcts.search()
print(f"Best answer: {best_answer}")
```

## License

This project is open source and available under the [MIT License](LICENSE).
