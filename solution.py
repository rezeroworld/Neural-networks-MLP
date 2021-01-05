import pickle
import numpy as np


class NN(object):
    def __init__(self,
                 hidden_dims=(512, 256),
                 datapath='cifar10.pkl',
                 n_classes=10,
                 epsilon=1e-6,
                 lr=7e-4,
                 batch_size=1000,
                 seed=None,
                 activation="relu",
                 init_method="glorot",
                 normalization=False
                 ):

        self.hidden_dims = hidden_dims
        self.n_hidden = len(hidden_dims)
        self.datapath = datapath
        self.n_classes = n_classes
        self.lr = lr
        self.batch_size = batch_size
        self.init_method = init_method
        self.seed = seed
        self.activation_str = activation
        self.epsilon = epsilon

        self.train_logs = {'train_accuracy': [], 'validation_accuracy': [], 'train_loss': [], 'validation_loss': []}

        if datapath is not None:
            u = pickle._Unpickler(open(datapath, 'rb'))
            u.encoding = 'latin1'
            self.train, self.valid, self.test = u.load()
            if normalization:
                self.normalize()
        else:
            self.train, self.valid, self.test = None, None, None

    def initialize_weights(self, dims):
        if self.seed is not None:
            np.random.seed(self.seed)

        self.weights = {}
        # self.weights is a dictionnary with keys W1, b1, W2, b2, ..., Wm, Bm where m - 1 is the number of hidden layers
        all_dims = [dims[0]] + list(self.hidden_dims) + [dims[1]]
        for layer_n in range(1, self.n_hidden + 2):
            # WRITE CODE HERE
            std = np.sqrt(6/(all_dims[layer_n-1] + all_dims[layer_n]))
            self.weights[f"W{layer_n}"] = np.random.uniform(low=-std, high=std, size=(all_dims[layer_n-1], all_dims[layer_n]) )
            self.weights[f"b{layer_n}"] = np.zeros((1, all_dims[layer_n]))

    def relu(self, x, grad=False):
        if grad:
            return (x > 0) * 1
        else:
            return (x > 0) * x

    def sigmoid(self, x, grad=False):
        if grad:
            return np.exp(-1 * x)/(1 + np.exp(-1 * x))**2
        else:
            return 1 / (1 + np.exp(-1 * x))

    def tanh(self, x, grad=False):
        if grad:
            return 1 - self.tanh(x) ** 2
        else:
            return (np.exp(x) - np.exp(-1 * x)) / (np.exp(x) + np.exp(-1 * x))

    def leakyrelu(self, x, grad=False):
        alpha = 0.01
        if grad:
            return (x > 0) * 1 + (x <= 0) * alpha
        else:
            return (x > 0)  * x +  (x <= 0)  * alpha * x

    def activation(self, x, grad=False):
        if self.activation_str == "relu":
            return self.relu(x, grad)
        elif self.activation_str == "sigmoid":
            return self.sigmoid(x, grad)
        elif self.activation_str == "tanh":
            return self.tanh(x, grad)
        elif self.activation_str == "leakyrelu":
            return self.leakyrelu(x, grad)
        else:
            raise Exception('invalid activation')

    def softmax(self, x):
        # Remember that softmax(x-C) = softmax(x) when C is a constant.
        if x.ndim == 1:
            return self.softmax_vec(x)
        else:
            return np.apply_along_axis(self.softmax_vec, 1, x)

    def softmax_vec(self, x):
        x = x - np.max(x)
        return np.exp(x) / np.exp(x).sum()

    def forward(self, x):
        cache = {"Z0": x}
        # cache is a dictionnary with keys Z0, A0, ..., Zm, Am where m - 1 is the number of hidden layers
        # Ai corresponds to the preactivation at layer i, Zi corresponds to the activation at layer i
        for layer_n in range(1, self.n_hidden + 2):
            cache[f'A{layer_n}'] = np.dot(cache[f'Z{layer_n - 1}'], self.weights[f'W{layer_n}']) + self.weights[f'b{layer_n}']
            if layer_n == self.n_hidden + 1: # last layer
                cache[f'Z{layer_n}'] = self.softmax(cache[f'A{layer_n}'])
            else:
                cache[f'Z{layer_n}'] = self.activation(cache[f'A{layer_n}'])

        return cache
    
    def backward(self, cache, labels):
        output = cache[f"Z{self.n_hidden + 1}"]
        grads = {}
        # grads is a dictionnary with keys dAm, dWm, dbm, dZ(m-1), dA(m-1), ..., dW1, db1
        
        grads[f"dA{self.n_hidden + 1}"] = output - labels
        
        if self.batch_size == 1:
            for l in range(self.n_hidden + 1, 0, -1):
                cache[f"Z{l-1}"] = cache[f"Z{l-1}"][np.newaxis, :]
                
        for l in range(self.n_hidden + 1, 1, -1):
            grads[f"dW{l}"] = np.dot(cache[f"Z{l-1}"].T, grads[f"dA{l}"])  / self.batch_size  
            grads[f"db{l}"] = np.sum(grads[f"dA{l}"], axis=0)[np.newaxis, :] / self.batch_size
            grads[f"dZ{l-1}"] = np.dot(grads[f"dA{l}"], self.weights[f"W{l}"].T)
            grads[f"dA{l-1}"] = np.multiply(grads[f"dZ{l-1}"], self.activation(cache[f"A{l-1}"], grad=True))
        grads["dW1"] = np.dot(cache["Z0"].T, grads["dA1"]) / self.batch_size
        grads["db1"] = np.sum(grads["dA1"], axis=0)[np.newaxis, :] / self.batch_size
            
        return grads

    def update(self, grads):
        for layer in range(1, self.n_hidden + 2):
            self.weights[f"W{layer}"] = self.weights[f"W{layer}"] - grads[f"dW{layer}"] * self.lr 
            self.weights[f"b{layer}"] = self.weights[f"b{layer}"] - grads[f"db{layer}"] * self.lr 

    def one_hot(self, y):
        y = y.astype(int)
        oh_labels = np.zeros((y.size, self.n_classes))
        oh_labels[np.arange(y.size), y] = 1
        return oh_labels

    def loss(self, prediction, labels):
        prediction[np.where(prediction < self.epsilon)] = self.epsilon
        prediction[np.where(prediction > 1 - self.epsilon)] = 1 - self.epsilon
        n = labels.shape[0]
        loss = 0
        for p, y in zip(prediction, labels):
            loss += self.cross_entropy(y, p)
        return loss / n

    def cross_entropy(self, y, p):
        return - np.sum(y * np.log(p))

    def compute_loss_and_accuracy(self, X, y):
        one_y = self.one_hot(y)
        cache = self.forward(X)
        predictions = np.argmax(cache[f"Z{self.n_hidden + 1}"], axis=1)
        accuracy = np.mean(y == predictions)
        loss = self.loss(cache[f"Z{self.n_hidden + 1}"], one_y)
        return loss, accuracy, predictions

    def train_loop(self, n_epochs):
        X_train, y_train = self.train
        X_valid, y_valid = self.valid
        y_onehot = self.one_hot(y_train)
        dims = [X_train.shape[1], y_onehot.shape[1]]
        self.initialize_weights(dims)

        n_batches = int(np.ceil(X_train.shape[0] / self.batch_size))

        for epoch in range(n_epochs):

            for batch in range(n_batches):
                minibatchX = X_train[self.batch_size * batch:self.batch_size * (batch + 1), :]
                minibatchY = y_onehot[self.batch_size * batch:self.batch_size * (batch + 1), :]
                cache = self.forward(minibatchX)
                grads = self.backward(cache, minibatchY)
                self.update(grads)

            train_loss, train_accuracy, _ = self.compute_loss_and_accuracy(X_train, y_train)
            valid_loss, valid_accuracy, _ = self.compute_loss_and_accuracy(X_valid, y_valid)

            print('epoch:', epoch, 'train: ', train_loss, train_accuracy, '; valid: ', valid_loss, valid_accuracy)

            self.train_logs['train_accuracy'].append(train_accuracy)
            self.train_logs['validation_accuracy'].append(valid_accuracy)
            self.train_logs['train_loss'].append(train_loss)
            self.train_logs['validation_loss'].append(valid_loss)

        return self.train_logs

    def evaluate(self):
        X_test, y_test = self.test
        test_loss, test_accuracy, _ = self.compute_loss_and_accuracy(X_test, y_test)
        return test_loss, test_accuracy

    def normalize(self):
        X_train, y_train = self.train
        X_valid, y_valid = self.valid
        X_test, y_test = self.test
        mean = X_train.mean()
        std = X_train.std()
        X_train = (X_train - mean) / std
        X_valid = (X_valid - mean) / std
        X_test = (X_test - mean) / std

        self.train = (X_train, y_train)
        self.valid = (X_valid, y_valid)
        self.test = (X_test, y_test)