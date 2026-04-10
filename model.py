import numpy as np

# Sample dataset (you can replace with your own)
X = np.array([[1], [2], [3], [4], [5]])   # features
y = np.array([2, 4, 6, 8, 10])            # labels

# Initialize parameters
w = 0.0   # weight
b = 0.0   # bias

# Hyperparameters
learning_rate = 0.01
epochs = 1000

n = len(X)

# Training loop
for i in range(epochs):
    y_pred = w * X + b

    # Compute loss (Mean Squared Error)
    loss = (1/n) * np.sum((y_pred - y)**2)

    # Compute gradients
    dw = (2/n) * np.sum((y_pred - y) * X)
    db = (2/n) * np.sum(y_pred - y)

    # Update parameters
    w = w - learning_rate * dw
    b = b - learning_rate * db

    if i % 100 == 0:
        print(f"Epoch {i}, Loss: {loss:.4f}")

print("\nFinal parameters:")
print("Weight:", w)
print("Bias:", b)

