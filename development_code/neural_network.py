import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.preprocessing import StandardScaler
import tqdm

# %matplotlib widget

from abc import ABC, abstractmethod

class Layer(ABC):
    """
    Abstract Base Class for all neural network layers.
    """

    def __init__(self):
        super().__init__()
        self.W: np.ndarray = None
        self.b: np.ndarray = None
        self.inp: Layer
        self.next: Layer
        self.out_dim: int
        self.inp_dim: int
        self.momment1: np.ndarray = None
        self.momment2: np.ndarray = None
        self.momment_b1: np.ndarray = None
        self.momment_b2: np.ndarray = None

    @abstractmethod
    def __call__(self, inp: Layer):
        """
        Forward pass of the layer.

        Args:
            inp: Input data.

        Returns:
            Output data.
        """
        pass

    @abstractmethod
    def forward(self, X: np.ndarray, y: np.ndarray = None) -> object:
        """
        Forward pass with NumPy arrays.

        Args:
            X: Input data.
            y: Target data (optional).

        Returns:
            Output data.
        """
        pass

    @abstractmethod
    def backward(self, dO: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Backward pass of the layer.

        Args:
            dO: Gradient of the loss function.

        Returns:
            Tuple containing gradient of the input, gradient of the weights, and gradient of the biases.
        """
        pass

    def __repr__(self):
        """
        String representation of the layer.
        """
        return self.__class__.__name__


class Dense(Layer):
    def __init__(self, out_dim, inp_dim=None, reg=0.001):
        self.W: np.ndarray = None
        self.b = np.random.normal(0, 0.2, out_dim)
        self.reg = reg
        self.out_dim = out_dim
        self.inp_dim = inp_dim
        self.next: Layer = None
        self.momment1 = None
        self.momment2 = None
        self.momment_b1 = np.zeros_like(self.b)
        self.momment_b2 = np.zeros_like(self.b)
        if inp_dim:
            # self.W = np.random.normal(0, 1, (self.out_dim, self.inp_dim))
            self.W = np.random.normal(0, np.sqrt(2.0/(self.out_dim + self.inp_dim)), (self.out_dim, self.inp_dim))
            self.momment1 = np.zeros_like(self.W)
            self.momment2 = np.zeros_like(self.W)

    def __call__(self, inp: Layer):
        self.inp = inp
        inp.next = self
        self.inp_dim = inp.out_dim
        # self.W = np.random.normal(0, 1, (self.out_dim, self.inp_dim))
        self.W = np.random.normal(0, np.sqrt(2.0/(self.out_dim + self.inp_dim)), (self.out_dim, self.inp_dim))
        self.momment1 = np.zeros_like(self.W)
        self.momment2 = np.zeros_like(self.W)
        return self
    
    def __repr__(self):
        return (self.__class__.__name__ + ' output: ' + str(self.out_dim) + ' input: ' + str(self.inp_dim))

    def forward(self, X, y=None, W=None, b=None):
        self.X = X
        if len(self.X.shape) > 2:
            X = X.reshape(-1, X.shape[-1])
        if not W:
            W = self.W
        if not b:
            b = self.b
        out = np.dot(W, X.T).T + b

        if len(self.X.shape) > 2:
            out = out.reshape(*self.X.shape[:-1], self.out_dim)
        return out # logits
    
    def backward(self, dO):
        dO_shape = dO.shape
        X = self.X
        if len(dO_shape) > 2 or len(self.X.shape) > 2:
            dO = dO.reshape(-1, dO.shape[-1])
            X = X.reshape(-1, self.X.shape[-1])
        
        dW = np.sum(X[:, np.newaxis, :] * dO[:, :, np.newaxis], axis=0) + self.reg*self.W
        dX = np.sum(self.W[:, :, np.newaxis] * dO.T[:, np.newaxis, :], axis=0).T
        db = np.sum(dO, axis=0)
        
        if len(self.X.shape) > 2:
            dX = dX.reshape(self.X.shape)
        return (dX, dW, db)
    
class LSTM(Layer):
    # inputs will include a mask
    # if return_seq is false, the last output will be determined using the mask
    def __init__(self, hidden_units, inp_dim=None, return_seq=True, return_mask=False, reg=0.001):
        self.W = None
        self.b = None
        self.reg = reg
        self.out_dim = hidden_units
        self.inp_dim = inp_dim
        self.next = None
        self.return_seq = return_seq
        self.return_mask = return_mask
        self.b = np.random.normal(0, 0.2, (4, self.out_dim))
        self.bf = self.b[0]
        self.bi = self.b[1]
        self.bo = self.b[2]
        self.bc = self.b[3]
        self.momment1 = None
        self.momment2 = None
        self.momment_b1 = np.zeros_like(self.b)
        self.momment_b2 = np.zeros_like(self.b)
        if inp_dim:
            self.W = np.random.normal(0, np.sqrt(2.0/(2 * self.out_dim + self.inp_dim)), (self.out_dim * 4, self.inp_dim + self.out_dim))
            self.Wf = self.W[:self.out_dim, :]
            self.Wi = self.W[self.out_dim: 2 * self.out_dim, :]
            self.Wo = self.W[2 * self.out_dim: 3 * self.out_dim, :]
            self.Wc = self.W[3 * self.out_dim:, :]
            self.b = np.random.normal(0, 0.2, 4 * self.out_dim)
            self.momment1 = np.zeros((self.out_dim * 4, self.inp_dim + self.out_dim))
            self.momment2 = np.zeros((self.out_dim * 4, self.inp_dim + self.out_dim))

    
    def __call__(self, inp: Layer):
        self.inp = inp
        inp.next = self
        self.inp_dim = inp.out_dim
        self.W = np.random.normal(0, np.sqrt(2.0/(2 * self.out_dim + self.inp_dim)), (self.out_dim * 4, self.inp_dim + self.out_dim))
        self.Wf = self.W[:self.out_dim, :]
        self.Wi = self.W[self.out_dim: 2 * self.out_dim, :]
        self.Wo = self.W[2 * self.out_dim: 3 * self.out_dim, :]
        self.Wc = self.W[3 * self.out_dim:, :]
        self.momment1 = np.zeros((self.out_dim * 4, self.inp_dim + self.out_dim))
        self.momment2 = np.zeros((self.out_dim * 4, self.inp_dim + self.out_dim))
        return self
    
    def __repr__(self):
        return (self.__class__.__name__ + ' output: ' + str(self.out_dim) + ' input: ' + str(self.inp_dim))
    
    @staticmethod
    def sigmoid(X):
        return 1.0/(1.0 + np.exp(-1 * X))
    
    def forward(self, X, y=None, W=None, b=None):
        #W = out x in
        self.Wf = self.W[:self.out_dim, :]
        self.Wi = self.W[self.out_dim: 2 * self.out_dim, :]
        self.Wo = self.W[2 * self.out_dim: 3 * self.out_dim, :]
        self.Wc = self.W[3 * self.out_dim:, :]

        self.bf = self.b[0]
        self.bi = self.b[1]
        self.bo = self.b[2]
        self.bc = self.b[3]

        if isinstance(X, dict):
            X_ = X
            X = X_['input_ids']
            mask = X_.get('seq_lens', np.array(X.shape[1]).repeat(X.shape[0]))
        else:
            mask = np.array(X.shape[1]).repeat(X.shape[0])
        fg = np.zeros((X.shape[0], X.shape[1] + 1, self.out_dim))
        ig = np.zeros((X.shape[0], X.shape[1] + 1, self.out_dim))
        og = np.zeros((X.shape[0], X.shape[1] + 1, self.out_dim))
        pstate = np.zeros((X.shape[0], X.shape[1] + 1, self.out_dim))
        state = np.zeros((X.shape[0], X.shape[1] + 1, self.out_dim))
        out = np.zeros((X.shape[0], X.shape[1] + 1, self.out_dim))
        X = np.concatenate([np.zeros((X.shape[0], 1, X.shape[2])), X], axis=1) # shape of X is (N, T, d)
        self.X = X
        self.mask = mask
        
        for t in range(1, X.shape[1]):
            fg[:, t, :] = self.sigmoid(np.dot(np.hstack([out[:, t - 1, :], X[:, t, :]]), self.Wf.T) + self.bf)
            ig[:, t, :] = self.sigmoid(np.dot(np.hstack([out[:, t - 1, :], X[:, t, :]]), self.Wi.T) + self.bi)
            pstate[:, t, :] = np.tanh(np.dot(np.hstack([out[:, t - 1, :], X[:, t, :]]), self.Wc.T) + self.bc)
            state[:, t, :] = fg[:, t, :] * state[:, t - 1, :] + ig[:, t, :] * pstate[:, t, :]
            og[:, t, :] = self.sigmoid(np.dot(np.hstack([out[:, t - 1, :], X[:, t, :]]), self.Wo.T) + self.bo)
            out[:, t, :] = og[:, t, :] * np.tanh(state[:, t, :])
        self.fg = fg
        self.ig = ig
        self.og = og
        self.pstate = pstate
        self.state = state
        self.out = out
        out = out[:, 1:, :] if self.return_seq else out[np.arange(X.shape[0]), mask, :]
        # return out[:, 1:, :] if self.return_seq else out[np.arange(X.shape[0]), mask, :]
        return {'input_ids': out, 'seq_lens': mask} if self.return_mask else out

    def backward(self, dO):
        if not self.return_seq:
            dO_ = np.zeros_like(self.out)
            dO_[np.arange(self.X.shape[0]), self.mask, :] = dO
            dO = dO_
        else:
            dO = np.concatenate([np.zeros((dO.shape[0], 1, dO.shape[2])), dO], axis=1)
        # print (dO)
        dstate = np.zeros_like(self.state)
        dWo = np.zeros_like(self.Wo)
        dWi = np.zeros_like(self.Wi)
        dWf = np.zeros_like(self.Wf)
        dWc = np.zeros_like(self.Wc)
        dbf = np.zeros_like(self.bf)
        dbi = np.zeros_like(self.bi)
        dbo = np.zeros_like(self.bo)
        dbc = np.zeros_like(self.bc)
        dX = np.zeros_like(self.X)
        mask = np.arange(self.X.shape[1]) > self.mask[:, np.newaxis]
        for t in range(self.X.shape[1] - 1, 0, -1):
            dstate[:, t, :] += dO[:, t, :] * self.og[:, t, :] * (1 - np.square(np.tanh(self.state[:, t, :])))
            dstate[:, t - 1, :] = dstate[:, t, :] * self.fg[:, t, :]
            
            dWo += np.dot((dO[:, t, :] * np.tanh(self.state[:, t, :]) * self.og[:, t, :] * (1 - self.og[:, t, :])).T, 
                          np.hstack([self.out[:, t, :], self.X[:, t, :]]))
            dWi += np.dot((dstate[:, t, :] * self.pstate[:, t, :] * self.ig[:, t, :] * (1 - self.ig[:, t, :])).T, 
                          np.hstack([self.out[:, t, :], self.X[:, t, :]]))
            dWf += np.dot((dstate[:, t, :] * self.state[:, t - 1, :] * self.fg[:, t, :] * (1 - self.fg[:, t, :])).T, 
                          np.hstack([self.out[:, t, :], self.X[:, t, :]]))
            dWc += np.dot((dstate[:, t, :] * self.ig[:, t, :] * (1 - np.square(self.pstate[:, t, :]))).T,
                          np.hstack([self.out[:, t, :], self.X[:, t, :]]))
            
            dO[:, t - 1, :] += np.dot(dstate[:, t, :] * self.ig[:, t, :] * (1 - np.square(self.pstate[:, t, :])), self.Wc[:, :self.out_dim])\
                            + np.dot(dstate[:, t, :] * self.state[:, t - 1, :] * self.fg[:, t, :] * (1 - self.fg[:, t, :]), self.Wf[:, :self.out_dim])\
                            + np.dot(dstate[:, t, :] * self.pstate[:, t, :] * self.ig[:, t, :] * (1 - self.ig[:, t, :]), self.Wi[:, :self.out_dim])\
                            + np.dot(dO[:, t, :] * np.tanh(self.state[:, t, :]) * self.og[:, t, :] * (1 - self.og[:, t, :]), self.Wo[:, :self.out_dim])
            
            dX[:, t, :] = np.dot(dstate[:, t, :] * self.ig[:, t, :] * (1 - np.square(self.pstate[:, t, :])), self.Wc[:, self.out_dim:])\
                        + np.dot(dstate[:, t, :] * self.state[:, t - 1, :] * self.fg[:, t, :] * (1 - self.fg[:, t, :]), self.Wf[:, self.out_dim:])\
                        + np.dot(dstate[:, t, :] * self.pstate[:, t, :] * self.ig[:, t, :] * (1 - self.ig[:, t, :]), self.Wi[:, self.out_dim:])\
                        + np.dot(dO[:, t, :] * np.tanh(self.state[:, t, :]) * self.og[:, t, :] * (1 - self.og[:, t, :]), self.Wo[:, self.out_dim:])

            dbo += (dO[:, t, :] * np.tanh(self.state[:, t, :]) * self.og[:, t, :] * (1 - self.og[:, t, :])).sum(axis=0)
            dbi += (dstate[:, t, :] * self.pstate[:, t, :] * self.ig[:, t, :] * (1 - self.ig[:, t, :])).sum(axis=0)
            dbf += (dstate[:, t, :] * self.state[:, t - 1, :] * self.fg[:, t, :] * (1 - self.fg[:, t, :])).sum(axis=0)
            dbc += (dstate[:, t, :] * self.ig[:, t, :] * (1 - np.square(self.pstate[:, t, :]))).sum(axis=0)

        dW = np.vstack([dWf, dWi, dWo, dWc]) + self.reg * self.W
        db = np.vstack([dbf, dbi, dbo, dbc])
        # print ('dX')
        # print (dX)
        # print ('dstate')
        # print (dstate)
        dX[mask, :] = 0
        return (dX[:, 1:, :], dW, db)
    
class Embeddings(Layer):
    def __init__(self, num_embeddings, embedding_dim, pad_idx=None, trainable=True, reg=0.0001):
        self.W = None
        self.b = 0.
        self.reg = reg
        self.out_dim = embedding_dim
        self.inp_dim = num_embeddings
        self.pad_idx = pad_idx
        self.trainable = trainable
        self.next = None
        self.momment1 = None
        self.momment2 = None
        if num_embeddings:
            self.W = np.random.normal(0, 1, (self.out_dim, self.inp_dim))
            self.momment1 = np.zeros_like(self.W)
            self.momment2 = np.zeros_like(self.W)

    def __call__(self, inp: Layer):
        self.inp = inp
        inp.next = self
        if self.inp_dim != inp.out_dim:
            raise ValueError('num_embeddings do not match out_dim of inputs')
        return self
    
    def __repr__(self):
        return (self.__class__.__name__ + ' embedding_dim: ' + str(self.out_dim) + ' num_embeddings: ' + str(self.inp_dim))
    
    def forward(self, X, y=None):
        self.X = X
        # inputs are encoded tokens with token_ids
        # shape of X: (N, T)
        
        out = np.take(self.W.T, X, axis=0)
        # output shape: N, T, emb_dim
        if self.pad_idx is not None:
            mask = (X != self.pad_idx).sum(axis=1)
        return out if self.pad_idx is None else {'input_ids': out, 'seq_lens': mask}
    
    def backward(self, dO):
        dW = np.zeros_like(self.W)
        # dW[:, self.X] += dO
        if self.trainable:
            np.add.at(dW.T, np.s_[self.X, :], dO)
        # dW += self.reg * self.W
        return 0, dW, None

    
class Activation(Layer):
    def __init__(self, func='relu'): # options: relu, softmax_with_cat_cross_entropy (softmax)
        self.act_function = func
        self.next = None

    def __call__(self, inp: Layer):
        self.inp = inp
        inp.next = self
        self.inp_dim = inp.out_dim
        self.out_dim = self.inp_dim
        return self
    
    def __repr__(self):
        return (self.__class__.__name__ + ' ' + self.act_function + ' output: ' + str(self.out_dim) + ' input: ' + str(self.inp_dim))
    
    def forward(self, X, y=None):
        self.X = X
        self.y = y
        if self.act_function == 'relu':
            out = np.maximum(0, X)
            activations = out
        elif self.act_function == 'softmax':
            exps = np.exp(X - np.max(X, axis=1, keepdims=True))
            activations = exps / np.sum(exps, axis=1, keepdims=True)
            activations = np.where(activations > 1.0e-10, activations, 1.0e-10)
            out = np.mean(-1*np.sum(y * np.log(activations), axis=1))
        elif self.act_function == 'sigmoid_with_bin_cross_entropy':
            sig = 1/(1 + np.exp(-X))
            activations = sig
            activations = np.where(activations > 1.0e-10, activations, 1.0e-10)
            out = np.mean(-1*((y * np.log(activations)) + ((1 - y) * np.log(1 - activations))))
        elif self.act_function == 'sigmoid':
            activations = 1/(1 + np.exp(-X))
            activations = np.where(activations > 1.0e-7, activations, 1.0e-7)
            activations = np.where(activations < 1 - 1.0e-7, activations, 1 - 1.0e-7)
            out = activations
        self.activations = activations
        return out
    
    def predict(self, X, y=None):
        _ = self.forward(X, y)
        return self.activations
    
    def backward(self, dO=None):
        if self.act_function == 'relu':
            dX = np.where(self.X < 0, 0, 1) * dO
        elif self.act_function == 'softmax' or self.act_function == 'sigmoid_with_bin_cross_entropy':
            dX = self.activations - self.y
        elif self.act_function == 'sigmoid':
            dX = self.activations * (1 - self.activations) * dO
        return (dX, None, None)
    
class Loss(Layer):
    def __init__(self, loss_fn): # options: mse
        self.loss_function = loss_fn
        self.next = None

    def __call__(self, inp: Layer):
        self.inp = inp
        inp.next = self
        self.inp_dim = inp.out_dim
        self.out_dim = self.inp_dim
        return self
    
    def __repr__(self):
        return (self.__class__.__name__ + ' ' + self.loss_function + ' output: ' + str(self.out_dim) + ' input: ' + str(self.inp_dim))
    
    def forward(self, X, y):
        self.X = X
        self.y = y
        if self.loss_function == 'mse':
            loss = np.mean((X - y)**2)
        return loss
    
    def predict(self, X, y=None):
        return X
    
    def backward(self, dO=None):
        if self.loss_function == 'mse':
            dX = 2*(self.X - self.y)
        return (dX, None, None)

class Optimizer(object): # Adam implementation
    def __init__(self, lr=0.001, b1=0.9, b2=0.999):
        self.b1, self.b2 = b1, b2
        self.eps = 1e-8
        self.t = 1
        self.lr = lr
        self.loss = []

    def run_forward(self, input_layer: Layer, X, y):
        self.input_layer = input_layer
        layer = input_layer
        out = X
        while (layer):
            # print (layer)
            # print (out.shape)
            out = layer.forward(out, y)
            layer = layer.next
        loss = out
        return loss

    def optimize_step(self, out_layer, verbose=False):
        layer = out_layer 
        t = self.t
        dO = 1  
        lr = self.lr 
        while (layer):
            # print (layer)
            dO, dW, db = layer.backward(dO)
            if dW is not None:
                moment1 = (self.b1 * layer.momment1) + ((1 - self.b1) * dW)
                moment2 = (self.b2 * layer.momment2) + ((1 - self.b2) * dW**2)
                mt1_hat = moment1/(1 - self.b1**t)
                mt2_hat = moment2/(1 - self.b2**t)
                # W = layer.W - (lr * dW)
                W = layer.W - (lr * mt1_hat/(np.sqrt(mt2_hat) + self.eps))
                layer.W = W
                layer.momment1 = moment1
                layer.momment2 = moment2
                if verbose >= 10:
                    print (layer)
                    print ('dW:', dW)
                    print ("updated W:")
                    print (W)

            if db is not None:
                moment_b1 = (self.b1 * layer.momment_b1) + ((1 - self.b1) * db)
                moment_b2 = (self.b2 * layer.momment_b2) + ((1 - self.b2) * db**2)
                mt1_hat = moment_b1/(1 - self.b1**t)
                mt2_hat = moment_b2/(1 - self.b2**t)
                # b = layer.b - (lr * db)
                b = layer.b - (lr * mt1_hat/(np.sqrt(mt2_hat) + self.eps))
                layer.b = b
                layer.momment_b1 = moment_b1
                layer.momment_b2 = moment_b2
                if verbose >= 10:
                    print ('db:', db)
                    print ("updated b:", b)

            self.t = t + 1
            layer = layer.inp
        return self.t

    def train(self, input_layer: Layer, out_layer: Layer, 
              X, y, batch_size=None, 
              patience=20, epochs=None, 
              verbose=False, loss_tr_ep=1.0e-10,
              inputs_batched=False):

        self.input_layer = input_layer
        self.out_layer = out_layer
        patience = patience
        loss_tracker = []
        epoch = 0
        if not epochs:
            epochs = 1e10
        if not patience:
            patience = epochs + 1
        patience_remaining = patience
        
        if not isinstance(batch_size, int) and not inputs_batched:
            batch_size = X.shape[0]
        elif inputs_batched:
            batch_size = X[0].shape[0]
        
        num_batches = len(X) if inputs_batched else int(np.ceil(X.shape[0]/batch_size))
        
        print ("Using batch_size of", batch_size)
        while (patience_remaining > 0 and epochs > epoch):
            loss_tracker_epoch = []
            for i in tqdm.tqdm(range(num_batches), disable=not verbose):
                if inputs_batched: # X is list of precreated batched
                    X_batch = X[i]
                    y_batch = y[i]
                else: # X is the entire data and batches are created here
                    up_ind = min(X.shape[0], (i + 1) * batch_size)
                    X_batch = X[i * batch_size: up_ind]
                    y_batch = y[i * batch_size: up_ind]
                loss = self.run_forward(input_layer, X_batch, y_batch)
                timestep = self.optimize_step(out_layer, verbose=verbose)
                loss_tracker_epoch.append(loss)
            epoch_loss = np.mean(loss_tracker_epoch)
            epoch += 1
            
            if len(loss_tracker) > 0 and epoch_loss + loss_tr_ep > min(loss_tracker):
                patience_remaining -= 1
            else:
                patience_remaining = patience
            loss_tracker.append(epoch_loss)

            print ('epoch:', epoch, 'loss:', epoch_loss)
        self.loss = self.loss + loss_tracker
    
    def predict(self, input_layer, X, y, batch_size=None, verbose=False):
        out_list = []
        if not isinstance(batch_size, int):
            batch_size = X.shape[0]
        print ("Using batch_size of", batch_size)
        for i in range(int(np.ceil(X.shape[0]/batch_size))):
            up_ind = min(X.shape[0], (i + 1) * batch_size)
            X_batch = X[i * batch_size: up_ind]
            y_batch = y[i * batch_size: up_ind]
            layer = input_layer
            out = X_batch
            while (layer):
                # print (layer)
                # print (out.shape)
                if isinstance(layer, (Activation, Loss)):
                # if isinstance(layer, relu1.__class__):
                    out = layer.predict(out, y_batch)
                else:
                    out = layer.forward(out, y_batch)
                if verbose:
                    print (layer)
                    print (out)
                layer = layer.next
            out_list.append(out)
        return np.vstack(out_list)
