

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv

from matplotlib import pyplot as plt



def init_data():
    data=pd.read_csv('/kaggle/input/digit-recognizer/train.csv') # actual path to the MNSIT Dataset


    data.head()

    data=np.array(data)

    print(data)

    np.random.shuffle(data)

    m,n=data.shape
    data_dev=data[0:1000].T

    Y_dev=data_dev[0]
    X_dev=data_dev[1:n]

    data_train=data[1000:m].T

    Y_train=data_train[0]
    X_train=data_train[1:n]


    print(X_train[:,0].shape)

    X_train=X_train/255
    X_dev=X_dev/255

    return X_train,Y_train,X_dev,Y_dev


def init_params(n_input=784, n_hidden=10, n_output=10):
    """
    Initialize weights (small random) and biases (zeros).
      - W1: (n_hidden, n_input)
      - b1: (n_hidden, 1)
      - W2: (n_output, n_hidden)
      - b2: (n_output, 1)
    """
    W1 = np.random.randn(n_hidden, n_input) * 0.01
    b1 = np.zeros((n_hidden, 1))
    W2 = np.random.randn(n_output, n_hidden) * 0.01
    b2 = np.zeros((n_output, 1))
    return W1, b1, W2, b2


def ReLU(Z):
    return np.maximum(Z, 0)


def deriv_ReLU(Z):
    return (Z > 0).astype(float)


def SoftMax(Z):
    """
    Z: shape (n_output, m)
    Returns A2 of shape (n_output, m), where each column is a softmax.
    """
    Z_shift = Z - np.max(Z, axis=0, keepdims=True)  # for numerical stability
    eZ = np.exp(Z_shift)
    return eZ / np.sum(eZ, axis=0, keepdims=True)


def forward_prop(W1, b1, W2, b2, X):
    """
    X: shape (n_input, m)
    Returns Z1, A1, Z2, A2:
      Z1 = W1@X + b1    → (n_hidden, m)
      A1 = ReLU(Z1)     → (n_hidden, m)
      Z2 = W2@A1 + b2   → (n_output, m)
      A2 = SoftMax(Z2)  → (n_output, m)
    """
    Z1 = W1 @ X + b1
    A1 = ReLU(Z1)
    Z2 = W2 @ A1 + b2
    A2 = SoftMax(Z2)
    return Z1, A1, Z2, A2


def one_hot(Y, n_output=10):
    """
    Y: 1D array of length m, labels in [0..n_output-1]
    Returns: (n_output, m) one-hot matrix.
    """
    m = Y.size
    one_hot_Y = np.zeros((n_output, m))
    return one_hot_Y


def backward_prop(Z1, A1, Z2, A2, W2, X, Y):
    """
    Z1, A1: (n_hidden, m)
    Z2, A2: (n_output, m)
    W2:     (n_output, n_hidden)
    X:      (n_input, m)
    Y:      1D array length m
    Returns dW1, db1, dW2, db2 with correct shapes.
    """
    m = Y.size


    n_output_val = A2.shape[0] #

    Y_oh = one_hot(Y, n_output=n_output_val)  # (n_output, m)
    dZ2 = A2 - Y_oh                           # (n_output, m)

    dW2 = (1/m) * (dZ2 @ A1.T)                # (n_output, n_hidden)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)  # (n_output, 1)

    dA1 = W2.T @ dZ2                          # (n_hidden, m)
    dZ1 = dA1 * deriv_ReLU(Z1)                # (n_hidden, m)

    dW1 = (1/m) * (dZ1 @ X.T)                 # (n_hidden, n_input)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)  # (n_hidden, 1)

    return dW1, db1, dW2, db2


def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    """
    Standard gradient‐descent update.
    """
    W1 = W1 - alpha * dW1
    b1 = b1 - alpha * db1
    W2 = W2 - alpha * dW2
    b2 = b2 - alpha * db2
    return W1, b1, W2, b2


def get_predictions(A2):
    """
    A2: (n_output, m)
    Returns a length‐m array of predicted class indices.
    """
    return np.argmax(A2, axis=0)


def get_accuracy(preds, Y):
    return np.mean(preds == Y)


def gradient_descent(X, Y, n_hidden=64, iterations=1000, alpha=0.01):
    """
    Trains a two‐layer net on (X, Y). Returns final W1,b1,W2,b2.
    X: shape (n_input, m)  (e.g. (784, m) for MNIST)
    Y: 1D array length m, labels in [0..k-1] for k classes

    - n_hidden: number of ReLU units in the hidden layer
    - iterations: how many full‐batch updates to do
    - alpha: learning rate
    """
    if Y.size == 0:
        raise ValueError("Cannot train with empty Y (labels). Ensure Y is not empty.")
    if X.shape[1] != Y.size:
        raise ValueError(f"Mismatch in number of samples: X has {X.shape[1]}, Y has {Y.size}.")

    n_input = X.shape[0]
    n_output = int(np.max(Y)) + 1

    W1, b1, W2, b2 = init_params(n_input=n_input,
                                  n_hidden=n_hidden,
                                  n_output=n_output)

    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)

        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W2, X, Y)

        W1, b1, W2, b2 = update_params(W1, b1, W2, b2,
                                       dW1, db1, dW2, db2,
                                       alpha)

        if i % 50 == 0:
            preds = get_predictions(A2)
            acc = get_accuracy(preds, Y)
            print(f"Iteration {i:4d} — training accuracy: {acc*100:.2f}%")

    return W1, b1, W2, b2


def get_validation_accuracy(W1,b1,W2,b2,X_dev,Y_dev):
    Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X_dev)
    preds = get_predictions(A2)
    acc = get_accuracy(preds, Y_dev)
    print(f"Validation accuracy: {acc*100:.2f}%")

if __name__=="__main__": 
    X_train,Y_train,X_dev,Y_dev=init_data()  
    W1, b1, W2, b2 = gradient_descent(X_train, Y_train)
    get_validation_accuracy(W1,b1,W2,b2,X_dev,Y_dev)
