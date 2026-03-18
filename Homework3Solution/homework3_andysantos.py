"""
CS 541 Homework 3
Andy Santos

 -> not a template <-
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize

NUM_HIDDEN_LAYERS = 3
NUM_INPUT = 784
NUM_HIDDEN = NUM_HIDDEN_LAYERS * [64]
NUM_OUTPUT = 10

# activation function
def relu(x):
    return np.maximum(0, x)

def softmax(z):
    # Numerically stable softmax
    z_shifted = z - np.max(z, axis=0, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)


def unpack(weights):
    # Unpack arguments
    Ws = []

    # Weight matrices
    start = 0
    end = NUM_INPUT * NUM_HIDDEN[0]
    W = weights[start:end]
    Ws.append(W)

    # Unpack the weight matrices as vectors
    for i in range(NUM_HIDDEN_LAYERS - 1):
        start = end
        end = end + NUM_HIDDEN[i] * NUM_HIDDEN[i + 1]
        W = weights[start:end]
        Ws.append(W)

    start = end
    end = end + NUM_HIDDEN[-1] * NUM_OUTPUT
    W = weights[start:end]
    Ws.append(W)

    # Reshape the weight "vectors" into proper matrices
    Ws[0] = Ws[0].reshape(NUM_HIDDEN[0], NUM_INPUT)
    for i in range(1, NUM_HIDDEN_LAYERS):
        Ws[i] = Ws[i].reshape(NUM_HIDDEN[i], NUM_HIDDEN[i - 1])
    Ws[-1] = Ws[-1].reshape(NUM_OUTPUT, NUM_HIDDEN[-1])

    # Bias terms
    bs = []
    start = end
    end = end + NUM_HIDDEN[0]
    b = weights[start:end]
    bs.append(b)

    for i in range(NUM_HIDDEN_LAYERS - 1):
        start = end
        end = end + NUM_HIDDEN[i + 1]
        b = weights[start:end]
        bs.append(b)

    start = end
    end = end + NUM_OUTPUT
    b = weights[start:end]
    bs.append(b)

    return Ws, bs

# Forward pass computing the mean cross-entropy loss.
def fCE(X, Y, weights):

    Ws, bs = unpack(weights)
    h = X
    for i in range(NUM_HIDDEN_LAYERS):
        z = Ws[i].dot(h) + np.atleast_2d(bs[i]).T
        h = relu(z)
    yhat = softmax(Ws[-1].dot(h) + np.atleast_2d(bs[-1]).T)
    n = X.shape[1]
    ce = -np.sum(Y * np.log(yhat + 1e-15)) / n
    return ce


# Computes the gradient of the mean CE loss w.r.t. all weights and biases using backpropagation.
def gradCE(X, Y, weights):

    Ws, bs = unpack(weights)
    n = X.shape[1]

    # Forward pass: store zs and hs
    zs = []
    hs = [X]
    h = X
    for i in range(NUM_HIDDEN_LAYERS):
        z = Ws[i].dot(h) + np.atleast_2d(bs[i]).T
        zs.append(z)
        h = relu(z)
        hs.append(h)
    yhat = softmax(Ws[-1].dot(h) + np.atleast_2d(bs[-1]).T)

    # Backward pass
    # Allocate gradient arrays
    dJdWs = [None] * (NUM_HIDDEN_LAYERS + 1)
    dJdbs = [None] * (NUM_HIDDEN_LAYERS + 1)

    # Starting gradient at output: d(CE)/d(z_output) = (yhat - Y) / n
    g = (yhat - Y) / n

    # Loop from output layer (index NUM_HIDDEN_LAYERS) back to input layer (index 0)
    for i in range(NUM_HIDDEN_LAYERS, -1, -1):
        dJdbs[i] = np.sum(g, axis=1)
        dJdWs[i] = g.dot(hs[i].T)
        g = Ws[i].T.dot(g)
        if i > 0:
            g = g * (zs[i - 1] > 0)  # relu derivative

    return np.hstack([dJdW.flatten() for dJdW in dJdWs] +
                     [dJdb.flatten() for dJdb in dJdbs])


# Rruns the forward pass and returns yhat.
def predict(X, weights):
    Ws, bs = unpack(weights)
    h = X
    for i in range(NUM_HIDDEN_LAYERS):
        z = Ws[i].dot(h) + np.atleast_2d(bs[i]).T
        h = relu(z)
    yhat = softmax(Ws[-1].dot(h) + np.atleast_2d(bs[-1]).T)
    return yhat


# Visualizes the first-layer weight matrix W(1).
def show_W0(W):
    Ws, bs = unpack(W)
    W = Ws[0]
    n = int(NUM_HIDDEN[0] ** 0.5)
    plt.imshow(np.vstack([
        np.hstack([np.pad(np.reshape(W[idx1 * n + idx2, :], [28, 28]), 2, mode='constant') for idx2 in range(n)])
        for idx1 in range(n)
    ]), cmap='gray')
    plt.title("First-layer weights W(1)")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig("W0_visualization.png", dpi=150)
    plt.show()


def train(trainX, trainY, weights, testX, testY, lr=5e-2):

    NUM_EPOCHS   = 60
    BATCH_SIZE   = 128
    LAMBDA       = 1e-4    # L2 regularization strength (weights only, not biases)
    LR_DECAY     = 0.97    # multiply lr by this factor each epoch
    NOISE_STD    = 0.05    # Gaussian noise for data augmentation

    n_train = trainX.shape[1]

    for epoch in range(NUM_EPOCHS):
        # Shuffle training data each epoch
        perm = np.random.permutation(n_train)
        Xshuf = trainX[:, perm]
        Yshuf = trainY[:, perm]

        for j in range(0, n_train, BATCH_SIZE):
            end = min(j + BATCH_SIZE, n_train)
            Xb = Xshuf[:, j:end].copy()
            Yb = Yshuf[:, j:end]

            # Data augmentation: add Gaussian noise
            Xb += np.random.randn(*Xb.shape) * NOISE_STD

            # Compute gradient
            grad = gradCE(Xb, Yb, weights)

            # L2 regularization gradient (weights only, not biases)
            Ws, bs = unpack(weights)
            reg_grad = np.hstack(
                [LAMBDA * W.flatten() for W in Ws] +
                [np.zeros_like(b) for b in bs]
            )

            weights -= lr * (grad + reg_grad)

        # Learning rate decay
        lr *= LR_DECAY

        # Evaluate on test set
        yhat = predict(testX, weights)
        ce = -np.sum(testY * np.log(yhat + 1e-15)) / testX.shape[1]
        acc = np.mean(np.argmax(yhat, axis=0) == np.argmax(testY, axis=0))
        print(f"Epoch {epoch + 1:3d}: test CE = {ce:.4f}, test accuracy = {acc * 100:.2f}%")

    return weights


def initWeightsAndBiases():
    Ws = []
    bs = []

    np.random.seed(0)
    W = 2 * (np.random.random(size=(NUM_HIDDEN[0], NUM_INPUT)) / NUM_INPUT ** 0.5) - 1. / NUM_INPUT ** 0.5
    Ws.append(W)
    b = 0.01 * np.ones(NUM_HIDDEN[0])
    bs.append(b)

    for i in range(NUM_HIDDEN_LAYERS - 1):
        W = 2 * (np.random.random(size=(NUM_HIDDEN[i + 1], NUM_HIDDEN[i])) / NUM_HIDDEN[i] ** 0.5) - 1. / NUM_HIDDEN[i] ** 0.5
        Ws.append(W)
        b = 0.01 * np.ones(NUM_HIDDEN[i + 1])
        bs.append(b)

    W = 2 * (np.random.random(size=(NUM_OUTPUT, NUM_HIDDEN[-1])) / NUM_HIDDEN[-1] ** 0.5) - 1. / NUM_HIDDEN[-1] ** 0.5
    Ws.append(W)
    b = 0.01 * np.ones(NUM_OUTPUT)
    bs.append(b)
    return Ws, bs


if __name__ == "__main__":
    # Load training data.
    # Divide pixels by 255 (range [0,1]), then subtract 0.5 (range [-0.5,+0.5]).
    from tensorflow.keras.datasets import fashion_mnist

    (trainX_raw, trainY_raw), (testX_raw, testY_raw) = fashion_mnist.load_data()
    trainX = trainX_raw.reshape(60000, 784).T / 255. - 0.5
    trainY = np.eye(NUM_OUTPUT)[trainY_raw].T
    testX = testX_raw.reshape(10000, 784).T / 255. - 0.5
    testY = np.eye(NUM_OUTPUT)[testY_raw].T

    Ws, bs = initWeightsAndBiases()

    # "Pack" all the weight matrices and bias vectors into long one parameter "vector".
    weights = np.hstack([W.flatten() for W in Ws] + [b.flatten() for b in bs])

    # On just the first 5 training examples, do numeric gradient check.
    print(scipy.optimize.check_grad(
        lambda weights_: fCE(np.atleast_2d(trainX[:, 0:5]), np.atleast_2d(trainY[:, 0:5]), weights_), \
        lambda weights_: gradCE(np.atleast_2d(trainX[:, 0:5]), np.atleast_2d(trainY[:, 0:5]), weights_), \
        weights))

    weights = train(trainX, trainY, weights, testX, testY, 0.05)
    show_W0(weights)