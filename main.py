import numpy as np
from sklearn.datasets import fetch_openml

def load_mnist_dataset():
    """
    Used for loading the mnist dataset.
    """
    mnist = fetch_openml('mnist_784', version=1)
    X = mnist.data / 255.0
    y = mnist.target.astype(int)
    
    # 60000 training examples and 10000 testing examples.
    train_x, train_y = X[:60000], y[:60000]
    test_x, test_y = X[60000:], y[60000:]
    
    return (train_x.to_numpy(), train_y.to_numpy()), (test_x.to_numpy(), test_y.to_numpy())

(train_x, train_y), (test_x, test_y) = load_mnist_dataset()

