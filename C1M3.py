# this program show a very simple classification example of multilayer NN
N = 500
eps = 1e-8
import numpy as np
class activation:
    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))

    @staticmethod
    def sigmoid_derivative(z):
        s = activation.sigmoid(z)
        return s * (1 - s)

    @staticmethod
    def relu(z):
        return np.maximum(0, z)

    @staticmethod
    def relu_derivative(z):
        return (z > 0).astype(float)
    
    @staticmethod
    def tanh(z):
        return np.tanh(z)
    
    @staticmethod
    def tanh_derivative(z):
        return 1 - np.tanh(z) ** 2

    @staticmethod
    def get(name):
        if name == 'sigmoid':
            return activation.sigmoid, activation.sigmoid_derivative, 1
        elif name == 'relu':
            return activation.relu, activation.relu_derivative, 1
        elif name == 'tanh':
            return activation.tanh, activation.tanh_derivative, 1
        else:
            raise ValueError(f"Unknown activation function: {name}")
        
class Layer:
    def __init__(self, input_dim, output_dim, activation_name='sigmoid'):
        self.activation, self.activation_deriv, initnum = activation.get(activation_name)
        self.W = np.random.randn(output_dim, input_dim) * np.sqrt(initnum / input_dim)
        self.b = np.random.randn(output_dim, 1) * np.sqrt(initnum / input_dim)
        self.Z = None
        self.A = None
        self.dW = None
        self.db = None

    def forward(self, A_prev):
        self.Z = np.dot(self.W, A_prev) + self.b  # shape: (output_dim, m)
        self.A = self.activation(self.Z)
        return self.A

    def backprop(self, dA, A_prev):
        m = A_prev.shape[1]
        dZ = dA * self.activation_deriv(self.Z)  # shape: (output_dim, m)
        self.dW = (1 / m) * np.dot(dZ, A_prev.T)  # shape: (output_dim, input_dim)
        self.db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)  # shape: (output_dim, 1)
        dA_prev = np.dot(self.W.T, dZ)  # shape: (input_dim, m)
        return dA_prev

class NeuralNetwork:
    def __init__(self, layer_sizes, activation_name = 'sigmoid'):
        self.layers = []
        for i in range(0, len(layer_sizes) - 1):
            if i == len(layer_sizes) - 2:
                layer = Layer(layer_sizes[i], layer_sizes[i + 1], activation_name='sigmoid')  # 最后一层用 sigmoid
            else:
                layer = Layer(layer_sizes[i], layer_sizes[i + 1], activation_name)
            self.layers.append(layer)
        print("layers:", len(self.layers))
    
    def forward(self, X):
        self.input_data = X
        A = X.T
        for layer in self.layers:
            A = layer.forward(A)
        return A

    def backprop(self, Y):
        Y = Y.T  # shape: (1, m)
        A_final = self.layers[-1].A  # shape: (1, m)

        # To avoid division by 0
        dA = - (Y / (A_final + eps)) + ((1 - Y) / (1 - A_final + eps))  # shape: (1, m)

        for i in reversed(range(len(self.layers))):
            layer = self.layers[i]
            A_prev = self.layers[i - 1].A if i > 0 else self.input_data.T  # use input if first layer
            dA = layer.backprop(dA, A_prev)

    def update_parameters(self, learning_rate = 0.01):
        for layer in self.layers:
            layer.W -= learning_rate * layer.dW
            layer.b -= learning_rate * layer.db

    def train(self, X, Y, epochs=1000, learning_rate=0.01):
        for epoch in range(epochs):
            self.forward(X)
            self.backprop(Y)
            self.update_parameters(learning_rate)
            if epoch % 100 == 0:
                loss = -np.mean(Y * np.log(self.layers[-1].A) + (1 - Y) * np.log(1 - self.layers[-1].A))
                print(f'Epoch {epoch}, Loss: {loss:.4f}')
            if epoch % 100 == 0:
                predictions = (self.layers[-1].A > 0.5).astype(int).T
                accuracy = np.mean(predictions == Y)
                print(f"Epoch {epoch}, Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")



def testmy(task="moons"):
    X,Y = None, None
    import matplotlib.pyplot as plt
    if task == "moons":
        from sklearn.datasets import make_moons
        X, Y = make_moons(n_samples = N, noise = 0.2, random_state=0)
        Y = Y.reshape(-1, 1)
    elif task == "circles":
        from sklearn.datasets import make_classification
        X, Y = make_classification(n_samples=500, n_features=2, n_redundant=0,
                                n_informative=2, n_clusters_per_class=1,
                                class_sep=1.0, random_state=0)
        Y = Y.reshape(-1, 1)
    


    from sklearn.model_selection import train_test_split

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Input data distribution
    axs[0].scatter(X[:, 0], X[:, 1], c=Y[:, 0], cmap=plt.cm.Spectral)
    axs[0].set_title("Input data distribution")

    nn = NeuralNetwork(layer_sizes=[2, 20, 10, 1], activation_name='tanh')
    nn.train(X_train, Y_train, epochs=50000, learning_rate=0.005)

    A_test = nn.forward(X_test)
    predictions_test = (A_test > 0.5).astype(int).T
    test_accuracy = np.mean(predictions_test == Y_test)
    print("-" * 20)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    # Decision boundary
    h = 0.01
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    Z = nn.forward(grid).T
    Z = (Z > 0.5).astype(int).reshape(xx.shape)

    axs[1].contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.5)
    axs[1].scatter(X[:, 0], X[:, 1], c=Y[:, 0], cmap=plt.cm.Spectral, edgecolors='k')
    axs[1].set_title("Decision Boundary")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    testmy("circles")