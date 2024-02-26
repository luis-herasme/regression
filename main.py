import numpy as np

def sigmoid(x: float) -> float:
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x: float) -> float:
    return sigmoid(x) * (1 - sigmoid(x))

input_size = 3
hidden_size = 4
output_size = 2

# Random weights
W1 = np.random.randn(input_size, hidden_size)
W2 = np.random.randn(hidden_size, output_size)

# Random bias
b1 = np.random.randn(hidden_size)
b2 = np.random.randn(output_size)

# Input
X = np.random.randn(input_size)

# Output
y = np.array([1, 0])

# Forward pass
z1 = np.dot(X, W1) + b1
a1 = sigmoid(z1)
z2 = np.dot(a1, W2) + b2
y_hat = sigmoid(z2)

# Backward pass
# Loss function
loss = np.square(y_hat - y).sum()
print(loss)

# Derivative of loss function
dloss = 2 * (y_hat - y)

