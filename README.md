# 🧠 Neural Network From Scratch (NumPy)

This project implements a **fully connected neural network (MLP)** from scratch using **NumPy** only.  
The network is **dataset-agnostic** and can be applied to any numerical input data, but the main example in this repository demonstrates training and testing on the **MNIST dataset** for handwritten digit classification.

It includes **activation functions**, **forward and backward propagation**, **mini-batch training**, **accuracy evaluation**, and **model persistence** using `pickle`.

---

## 📚 Table of Contents

1. [🚀 Features](#-features)  
2. [🧩 Project Structure](#-project-structure)  
3. [⚙️ Example Usage](#️-example-usage)  
4. [🧮 Activation Functions](#-activation-functions)  
5. [💾 Model Persistence](#-model-persistence)  
6. [🧠 Network Architecture Overview](#-network-architecture-overview)  
7. [📈 Training Monitoring](#-training-monitoring)  
8. [🧪 Testing and Predictions](#-testing-and-predictions)  
9. [🧰 Requirements](#-requirements)  
10. [👨‍💻 Author](#-author)  
11. [📜 License](#-license)

---

## 🚀 Features

- Manual implementation of:
  - Forward and backward propagation  
  - Gradient computation  
  - Weight and bias updates using a learning rate  
- Built-in activation functions:
  - **ReLU**
  - **Softmax**
- Support for **mini-batch training**
- **Training and testing accuracy** evaluation
- **Model saving and loading** via `pickle`
- Clean **object-oriented** and extensible design

---

## 🧩 Project Structure

```yaml
Neural-Network-from-Scratch/
├── functions.py          # Activation functions and utilities
├── model.py              # Main MLP class implementation
├── train.py              # Training and testing script
├── requirements.txt      # Requirements for running the project
└── README.md             # This file
```

## ⚙️ Example Usage 
```python
from model import MLPModel

# Create a new model
model = MLPModel(
    activation_function="relu",
    learning_rate=0.01,
    layers=(64, 32, 10)  # Example: 64 inputs, 32 hidden neurons, 10 outputs
)

# Train the model
model.train_model(train_x, train_y, epochs=10, batch_size=32)

# Evaluate on test data
accuracy = model.test_accuracy(test_x, test_y)
print(f"Test accuracy: {accuracy*100:.2f}%")

# Save the trained model
model.save_model("model.pkl")

# Load an existing model
loaded_model = MLPModel.load_model("model.pkl")

# Make predictions
pred = loaded_model.predict(sample)
print("Prediction:", pred)
```

## 🧮 Activation Functions

Activation functions are implemented as subclasses of an abstract base class Activation_Function.

ReLU
```python
ReLu().activation(Z)
```

Applies the function max(0, Z) element-wise.

Softmax
```python
Softmax().activation(Z)
```

Applies the softmax function row-wise to transform outputs into probability distributions — typically used for multi-class classification.

## 💾 Model Persistence

Weights, biases, and hyperparameters are saved using Python’s built-in pickle module.

```python
# Save model
model.save_model("model.pkl")

# Load model
model = MLPModel.load_model("model.pkl")
```

⚠️ The loaded model must have the same structure (layers and activation function) as the one originally trained.

## 🧠 Network Architecture Overview

The network is a fully connected feedforward neural network with the following internal components:

    - `self.weights:` list of weight matrices

    - `self.biases`: list of bias vectors

Training involves:

1. Forward pass: compute pre-activations Z and activations A

2. Backward pass: propagate error gradients

3. Parameter update:

    W -= learning_rate * dW  
    b -= learning_rate * db

## 📈 Training Monitoring

During training, accuracy is printed for each epoch:

```python
Epoch 1/10 — Train Accuracy: 82.45%
Epoch 2/10 — Train Accuracy: 86.71%
...
```

## 🧪 Testing and Predictions

The test_accuracy method evaluates the model on test data and displays predictions:

```python
model.test_accuracy(test_x, test_y)
# Model predicted 7. Correct answer: 7
# Model predicted 2. Correct answer: 2
# ...
```

## 🧰 Requirements

Python 3.10+

Install dependencies:
```bash
pip install -r requirements.txt
```

## 👨‍💻 Author

Developed by Ricardo Santos — focused on understanding the fundamentals of neural networks and implementing AI algorithms from scratch.

## 📜 License

This project is open for educational and research purposes.
Feel free to modify, extend, or build upon it.
