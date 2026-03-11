import numpy as np

# XOR verisi
X = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
y = np.array([[0],
              [1],
              [1],
              [0]])

np.random.seed(0)
input_size = 2
hidden_size = 2
output_size = 1
lr = 0.1
epochs = 10000

W1 = np.random.randn(input_size, hidden_size)
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size)
b2 = np.zeros((1, output_size))


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_deriv(x):
    return x * (1 - x)


for _ in range(epochs):
    z1 = np.dot(X, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    y_hat = sigmoid(z2)

    error = y_hat - y
    d_y_hat = error * sigmoid_deriv(y_hat)

    dW2 = np.dot(a1.T, d_y_hat)
    db2 = np.sum(d_y_hat, axis=0, keepdims=True)

    d_a1 = np.dot(d_y_hat, W2.T)
    d_z1 = d_a1 * sigmoid_deriv(a1)
    dW1 = np.dot(X.T, d_z1)
    db1 = np.sum(d_z1, axis=0, keepdims=True)

    W2 -= lr * dW2
    b2 -= lr * db2
    W1 -= lr * dW1
    b1 -= lr * db1


print("Egitim sonrasi tahminler:")
for x in X:
    z1 = np.dot(x, W1) + b1
    a1 = sigmoid(z1)
    z2 = np.dot(a1, W2) + b2
    y_pred = sigmoid(z2)
    print(f"{x} -> {y_pred[0][0]:.4f} (yuvarlanmis: {int(y_pred[0][0] > 0.5)})")
