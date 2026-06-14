import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils import check_random_state

class CustomNeuralNetworkRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, layer_sizes, epochs=5, learning_rate=0.001, batch_size=64, print_every=1, random_state=None):
        self.layer_sizes = layer_sizes
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.print_every = print_every
        self.random_state = random_state
        self.weights = []
        self.biases = []

    def _initialize_parameters(self):
        self.weights = []
        self.biases = []
        for i in range(len(self.full_layer_sizes_) - 1):
            in_dim = self.full_layer_sizes_[i]
            out_dim = self.full_layer_sizes_[i + 1]
            W = self.rng_.normal(0, np.sqrt(2.0 / in_dim), size=(out_dim, in_dim))
            B = np.zeros((out_dim, 1))
            self.weights.append(W)
            self.biases.append(B)

    def _relu(self, preactivation):
        return preactivation.clip(0.0)

    def _indicator_function(self, x):
        return (np.array(x) > 0).astype(float)

    def _compute_network_output(self, net_input):
        L = len(self.weights)
        all_f = [None] * L
        all_h = [None] * (L + 1)
        all_h[0] = net_input
        
        for layer in range(L - 1):
            all_f[layer] = np.matmul(self.weights[layer], all_h[layer]) + self.biases[layer]
            all_h[layer + 1] = self._relu(all_f[layer])

        all_f[L - 1] = np.matmul(self.weights[L - 1], all_h[L - 1]) + self.biases[L - 1]
        return all_f[L - 1], all_f, all_h

    def _least_squares_loss(self, net_output, y):
        return np.mean(np.sum((net_output - y) ** 2, axis=0))

    def _d_loss_d_output(self, net_output, y):
        return 2 * (net_output - y)

    def _backward_pass(self, all_f, all_h, y):
        L = len(self.weights)
        current_batch_size = y.shape[1]
        all_dl_dweights = [None] * L
        all_dl_dbiases = [None] * L
        all_dl_df = [None] * L
        all_dl_dh = [None] * L

        all_dl_df[L - 1] = self._d_loss_d_output(all_f[L - 1], y)

        for layer in range(L - 1, -1, -1):
            all_dl_dbiases[layer] = np.sum(all_dl_df[layer], axis=1, keepdims=True) / current_batch_size
            all_dl_dweights[layer] = np.matmul(all_dl_df[layer], all_h[layer].T) / current_batch_size
            if layer > 0:
                all_dl_dh[layer] = np.matmul(self.weights[layer].T, all_dl_df[layer])
                all_dl_df[layer - 1] = all_dl_dh[layer] * self._indicator_function(all_f[layer - 1])

        return all_dl_dweights, all_dl_dbiases

    def fit(self, X, Y):
        X = np.asarray(X)
        Y = np.asarray(Y)
        self.rng_ = check_random_state(self.random_state)
        input_dim = X.shape[1]
        self.full_layer_sizes_ = [input_dim] + list(self.layer_sizes)
        self._initialize_parameters()
        n_samples = len(X)
        total_batches = int(np.ceil(n_samples / self.batch_size))
        
        print(f"Rozpoczęcie trenowania (architektura: {self.full_layer_sizes_}, batch_size: {self.batch_size})...")
        print(f"Liczba próbek: {n_samples} | Liczba batchy na epokę: {total_batches}\n")
        
        for epoch in range(self.epochs):
            indices = np.arange(n_samples)
            self.rng_.shuffle(indices)
            total_loss = 0
            num_batches = 0
            
            for i in range(0, n_samples, self.batch_size):
                batch_indices = indices[i:i + self.batch_size]
                X_batch = X[batch_indices]
                Y_batch = Y[batch_indices]
                x = X_batch.T
                y = Y_batch.reshape(1, -1)
                
                output, all_f, all_h = self._compute_network_output(x)
                loss = self._least_squares_loss(output, y)
                total_loss += loss
                num_batches += 1
                
                all_dl_dweights, all_dl_dbiases = self._backward_pass(all_f, all_h, y)
                clip_val = 1.0
                
                for layer in range(len(self.weights)):
                    all_dl_dweights[layer] = np.clip(all_dl_dweights[layer], -clip_val, clip_val)
                    all_dl_dbiases[layer] = np.clip(all_dl_dbiases[layer], -clip_val, clip_val)
                    
                for layer in range(len(self.weights)):
                    self.weights[layer] -= self.learning_rate * all_dl_dweights[layer]
                    self.biases[layer] -= self.learning_rate * all_dl_dbiases[layer]

                if epoch % self.print_every == 0:
                    print(f"\rEpoka {epoch}: Przetwarzanie batcha {num_batches}/{total_batches}", end="", flush=True)
            
            avg_loss = total_loss / num_batches
            if epoch % self.print_every == 0:
                print(f" | Gotowe! Średni Loss: {avg_loss:.6f}")
                
        return self

    def predict(self, X):
        X = np.asarray(X)
        x = X.T
        net_output, _, _ = self._compute_network_output(x)
        predictions = net_output[0, :]
        predictions = np.maximum(0.0, predictions)
        return predictions