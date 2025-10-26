import numpy as np
from sklearn.datasets import fetch_openml

from train import MLPModel

def load_mnist_dataset():
    """
    Used for loading the mnist dataset.
    -------
    Returns:
        Two tuples with the training examples and their labels
        and the same for testing examples.
    """
    mnist = fetch_openml('mnist_784', version=1)
    X = mnist.data / 255.0
    y = mnist.target.astype(int)
    
    # 60000 training examples and 10000 testing examples.
    train_x, train_y = X[:60000], y[:60000]
    test_x, test_y = X[60000:], y[60000:]
    
    return (train_x.to_numpy(), train_y.to_numpy()), (test_x.to_numpy(), test_y.to_numpy())


if __name__ == "__main__":
    # Train a new model
    (train_x, train_y), (test_x, test_y) = load_mnist_dataset()
    # model = MLPModel(layers= (784, 12, 10), activation_function= 'ReLu', learning_rate=0.5)
    # model.train_model(train_x, train_y, epochs=250, batch_size=64)
    # print(f"Final model accuracy: {model.test_accuracy(test_x, test_y)*100}")
    # model.save_model()

    # Below is an example of loading an already trained model. Uncomment for use.
    model: MLPModel = MLPModel.load_model()
    input_sample = 0
    prediction = model.predict(train_x[input_sample])
    print(f"Model predicted {prediction}. Correct answer {train_y[input_sample]}")



