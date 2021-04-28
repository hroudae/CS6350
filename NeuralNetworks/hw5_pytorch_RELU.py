import torch
from torch import nn
from torch.nn import Module, ReLU, ModuleList, Parameter
import pandas as pd
import numpy as np

class SelfLinear(Module):
    def __init__(self, n_in, n_out):
        super(SelfLinear, self).__init__()
        # Use Kaiming (aka He) intialization
        w = torch.empty(n_out, n_in)
        nn.init.kaiming_uniform_(w)
        self.weight = Parameter(w.double())
        b = torch.empty(1, n_out)
        nn.init.kaiming_uniform_(b)
        self.bias = Parameter(b.double())
    
    def forward(self, X):
        return X @ self.weight.T  + self.bias
    
class Net(Module):
    def __init__(self, layers):
        super(Net, self).__init__()
        self.act = ReLU()
        self.fcs = ModuleList()
        self.layers = layers
        
        for i in range(len(self.layers)-1):
            self.fcs.append(SelfLinear(self.layers[i], self.layers[i+1]))
            
    def forward(self, X):
        for fc in self.fcs[:-1]:
            X = fc(X)
            X = self.act(X)
        X = self.fcs[-1](X)
        return X



training_data = pd.read_csv('bank-note/train.csv')
test_data = pd.read_csv('bank-note/test.csv')

# Prepare data
x_train = training_data.to_numpy()
y_train = np.matrix(x_train[:,-1]).T
y_train = 2*y_train - 1
x_train = x_train[:,:-1]
# Augment a 1 for the bias
x_train = np.concatenate((x_train, np.ones(x_train.shape[0])[:,None]), axis=1)

x_test = test_data.to_numpy()
y_test = np.matrix(x_test[:,-1]).T
y_test = 2*y_test - 1
x_test = x_test[:,:-1]
# Augment a 1 for the bias
x_test = np.concatenate((x_test, np.ones(x_test.shape[0])[:,None]), axis=1)

# Convert the data to PyTorch tensors
x = torch.tensor(x_train)
y = torch.tensor(y_train)

x_test_tensor = torch.tensor(x_test)
y_test_tensor = torch.tensor(y_test)


depth = [3, 5, 9]
width = [5, 10, 25, 50, 100]

print()
print()
print("********** Part 3e **********")
print("Neural network with PyTorch using RELU activation function")

print(f"Depth\tWidth\tTrain Error\tTest Error")
for d in depth:
    for w in width:
        layers = [x.shape[1]]
        layers += ([w for i in range(d)])
        layers += [1]
        model = Net(layers)
        optimizer = torch.optim.Adam(model.parameters())

        # run for 100 epochs
        for epoch in range(100):
            optimizer.zero_grad()
            L = ((model(x) - y)**2).sum()
            L.backward()
            optimizer.step()

        # find the test and training error
        with torch.no_grad():
            y_train_pred = (model(x)).detach().numpy()
            y_test_pred = (model(x_test_tensor)).detach().numpy()
        y_train_pred[y_train_pred >= 0] = 1
        y_train_pred[y_train_pred < 0] = -1
        numWrong = np.sum(np.abs(y_train_pred-y_train) / 2)
        train_err = numWrong/y_train.shape[0]
        
        y_test_pred[y_test_pred >= 0] = 1
        y_test_pred[y_test_pred < 0] = -1
        numWrong = np.sum(np.abs(y_test_pred-y_test) / 2)
        test_err = numWrong/y_test.shape[0]

        print(f"{d}\t{w}\t{train_err:.7f}\t{test_err:.7f}")
