# Neural Network From Scratch

A complete implementation of a neural network from scratch using only NumPy, without any deep learning frameworks. This project is designed to help you understand the fundamental concepts of deep learning by building everything from the ground up.

## Features

- **Pure NumPy Implementation**: No deep learning frameworks used
- **Comprehensive Layer Support**:
  - Dense (Fully Connected)
  - Activation Functions (ReLU, Sigmoid, Tanh, Softmax, LeakyReLU)
  - Dropout
  - Batch Normalization
- **Loss Functions**:
  - Mean Squared Error (MSE)
  - Categorical Cross-Entropy
  - Binary Cross-Entropy
- **Optimizers**:
  - Stochastic Gradient Descent (SGD)
  - SGD with Momentum
  - RMSprop
  - Adam
- **Training Utilities**:
  - Learning rate scheduling
  - Early stopping
  - Model checkpointing
  - Training progress visualization

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Examples](#examples)
- [Performance](#performance)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Prerequisites

- Python 3.8+
- NumPy
- Matplotlib (for visualization)
- scikit-learn (for datasets and metrics)

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/neural-network-from-scratch.git
cd neural-network-from-scratch

# Install dependencies
pip install numpy matplotlib scikit-learn
```

## Quick Start

### Training a Simple Neural Network

```python
import numpy as np
from neural_network import NeuralNetwork
from layers import Dense, Activation
from losses import CategoricalCrossentropy
from optimizers import Adam
from utils import load_mnist, to_categorical

# Load and preprocess data
X_train, y_train, X_test, y_test = load_mnist()
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Create model
model = NeuralNetwork()
model.add(Dense(784, 128))
model.add(Activation('relu'))
model.add(Dense(128, 64))
model.add(Activation('relu'))
model.add(Dense(64, 10))
model.add(Activation('softmax'))

# Compile and train
model.compile(
    loss=CategoricalCrossentropy(),
    optimizer=Adam(learning_rate=0.001)
)

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=10,
    batch_size=32
)

# Evaluate
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.4f}")
```

## Architecture

### Core Components

1. **Layers**
   - `Dense`: Fully connected layer
   - `Activation`: Applies activation functions
   - `Dropout`: Implements dropout regularization
   - `BatchNormalization`: Normalizes layer inputs

2. **Loss Functions**
   - `MSE`: Mean Squared Error for regression
   - `CategoricalCrossentropy`: For multi-class classification
   - `BinaryCrossentropy`: For binary classification

3. **Optimizers**
   - `SGD`: Stochastic Gradient Descent
   - `Momentum`: SGD with momentum
   - `RMSprop`: Root Mean Square Propagation
   - `Adam`: Adaptive Moment Estimation

4. **Activation Functions**
   - ReLU, LeakyReLU
   - Sigmoid, Tanh
   - Softmax (for output layer)

## Examples

### 1. MNIST Classification

```bash
python examples/mnist_classification.py
```

### 2. Binary Classification

```bash
python examples/binary_classification.py
```

### 3. Regression

```bash
python examples/regression.py
```

## Performance

### MNIST Results
- **Model**: 784-128-64-10 (ReLU activation)
- **Optimizer**: Adam (learning_rate=0.001)
- **Batch Size**: 32
- **Epochs**: 20

| Metric       | Training | Validation |
|--------------|----------|------------|
| Accuracy     | 99.2%    | 97.8%      |
| Loss         | 0.023    | 0.078      |

## Best Practices

1. **Data Preprocessing**
   - Normalize inputs to [0, 1] or [-1, 1]
   - Use one-hot encoding for classification labels
   - Shuffle training data before each epoch

2. **Model Architecture**
   - Start with a simple architecture and gradually increase complexity
   - Use Batch Normalization for deeper networks
   - Apply Dropout for regularization

3. **Training**
   - Use learning rate scheduling
   - Monitor training and validation metrics
   - Implement early stopping

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/)
- [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.github.io/)
- [Deep Learning Book](https://www.deeplearningbook.org/)
