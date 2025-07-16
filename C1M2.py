'''
This serves as an example of logistic regression with a neural network mindset.
'''

import numpy as np
def sigmoid(z):
    # sigmoid function
    return 1 / (1 + np.exp(-z))
M = 100 # number of pics
N = 10 # pic size
train_size = 20
def createdata(size=M):
    # generate pics for following examples:
    # pic size: 10 * 10, 0/1 for black/white

    X = np.random.randint(0, 2, (size, N * N))  # M pics, each pic is a vector of size N*N
    # if the sum of a pic > 80, lable as 1 or 0
    Y = np.array([1 if np.sum(x) >= 60 and np.sum(x) <= 90 else 0 for x in X])  # labels based on the sum of pixels
    Y = Y.reshape((size, 1))  # reshape to M * 1
    print(Y.shape)
    return X, Y

def train(X, Y):
    # we train a NN with one layer and one neuron
    # X: M * N*N, Y: M * 1
    W = np.random.rand(N * N, 1)  # weights for the neuron
    b = np.random.rand(1)  # bias for the neuron
    learning_rate = 0.01  # learning rate
    epochs = 1000  # number of iterations for training
    for epoch in range(epochs):
        # forward pass
        z = np.dot(X, W) + b
        A = sigmoid(z)  # activation function
        # compute loss (binary cross-entropy)
        loss = -np.mean(Y * np.log(A) + (1 - Y) * np.log(1 - A))
        # backward pass
        dA = A - Y  # derivative of loss with respect to activation
        dW = np.dot(X.T, dA) / M  # derivative of loss
        db = np.sum(dA) / M
        # update weights and bias
        W -= learning_rate * dW
        b -= learning_rate * db
        if epoch % 100 == 0:
            print(f'Epoch {epoch}, Loss: {loss:.4f}')
    return W, b
if __name__ == '__main__':
    X, Y = createdata(size=M)
    W, b = train(X, Y)
    test_X, test_Y = createdata(size=train_size)
    # test the model
    test_z = np.dot(test_X, W) + b
    test_A = sigmoid(test_z)
    test_predictions = (test_A >= 0.5).astype(int)  # threshold
    accuracy = np.mean(test_predictions == test_Y)
    print(f'Test Accuracy: {accuracy:.2f}')
