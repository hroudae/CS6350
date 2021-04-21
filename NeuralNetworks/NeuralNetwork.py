##########
# Author: Evan Hrouda
# Purpose: Implement an artificiial neural network
#          Sigmoid is used as the activation function and square loss for the predictions
##########
import csv
from dataclasses import dataclass
import numpy as np

#####
# Author: Evan Hrouda
# Purpose: Set gamma schedule values
#####
@dataclass
class GammaSchedule:
    gamma0: float
    d: float

#####
# Author: Evan Hrouda
# Purpose: A neural network class to hold relevant informaiton about the network
#          layers is the number of layers
#          numInputs is the number of inputs from the training example
#          hiddenNodeCount is a list of integers representing the number of hidden
#                nodes in each layer
#          self.weights is a 3d matrix of weights where the main index is the layer number,
#                the secondary index is the node the weight vector is coming from,
#                and the third is node positon the weight vector is pointing to.
#                If randInit is false, it will be initialized to 0
#                If randInit is true, it will be randomly initialized from the standard
#                Gaussin distribution
#####
class NeuralNet:
    def __init__(self, layers, numInputs, hiddenNodeCount, randInit):
        self.layerCount = layers
        # number of nodes at each layer, output layer is 2 nodes since y is located at 1 and not 0
        self.layerNodeCounts = np.concatenate([np.array([numInputs]), np.array(hiddenNodeCount), np.array([2])])
        # create the matrix of nodes with layer 0 being input
        self.nodes = np.zeros((layers, np.amax(self.layerNodeCounts)))
        self.nodes[:,0] = np.ones(layers) # set first to 1
        # Layer count by maximum hidden layers matrix
        self.weights = np.zeros((layers, np.amax(self.layerNodeCounts), np.amax(self.layerNodeCounts)))
        if randInit == True:
            self.weights = np.random.normal(size=(layers,np.amax(self.layerNodeCounts),np.amax(self.layerNodeCounts)))
        # the derivative of the weights calculated during backpropagation
        self.dweights = np.zeros((layers,np.amax(self.layerNodeCounts),np.amax(self.layerNodeCounts)))
        # the last output calculated from forward pass
        self.y = None

#####
# Author: Evan Hrouda
# Purpose: Parse a csv file into a feature matrix x and label vector y
#####
def parseCSV(csvFilePath, zero2neg):
    x = []
    y = []

    with open(csvFilePath, 'r') as csvFile:
        csvReader = csv.reader(csvFile, delimiter=',')

        for row in csvReader:
            thisExample = [float(row[i]) for i in range(len(row)-1)]
            thisExample += [1.0]
            x.append(thisExample)
            # need to convert label 0 to -1
            if zero2neg:
                y.append(2*float(row[-1]) - 1)
            else:
                y.append(float(row[-1]))

    x = np.matrix(x)
    y = np.array(y)
    return x, y

#####
# Author: Evan Hrouda
# Purpose: Sigmoid function to use as an activation function
#####
def sigmoid(x):
    from math import exp
    return 1 / (1 + exp(-1*x))

#####
# Author: Evan Hrouda
# Purpose: Sigmoid derivative function to use during backpropagation
#          x should be the value of sigmoid already
#####
def sigmoid_deriv(x):
    return x * (1 - x)

#####
# Author: Evan Hrouda
# Purpose: Implement the back-propagation algorithm for the three layer architecture
#          described in homework 5
#####
def NeuralNetwork_Backpropagation(y, nn):
    # compute loss
    dLdy = nn.y - y
    cache = np.zeros((len(nn.layerNodeCounts), np.amax(nn.layerNodeCounts)))

    # reverse the number of nodes at each level since this is backwards
    for target in reversed(range(1, len(nn.layerNodeCounts))):
        if target != 0 and target == nn.layerCount: # output layer and more than one layer
            for to in range(1, nn.layerNodeCounts[target]):
                cache[target, to] = dLdy
                for fromNode in range(nn.layerNodeCounts[target-1]):
                    nn.dweights[target-1,to,fromNode] = cache[target, to] * nn.nodes[target-1, fromNode]
        else: # hidden layers and not the last layer
            # calculated the cached partials
            # for each of the nodes, sum the connected partials saved in the cache and calcute the new one
            for to in range(1, nn.layerNodeCounts[target]):
                cache[target, to] = 0
                for connected in range(1, nn.layerNodeCounts[target+1]):
                    cache[target, to] += cache[target+1, connected] * nn.weights[target, connected, to] * sigmoid_deriv(nn.nodes[target, to])
            # now calculate the weight derivatives
            for to in range(nn.layerNodeCounts[target]):
                for fromNode in range(nn.layerNodeCounts[target-1]):
                    nn.dweights[target-1,to,fromNode] = cache[target, to] * nn.nodes[target-1, fromNode]

#####
# Author: Evan Hrouda
# Purpose: Implement forward pass prediction for the architecture described in homework 5
#          Sigmoid is used as the activation function
#####
def NeuralNetwork_Forwardpass(x, nn):
    # set the input layer of NerualNet to current example
    nn.nodes[0,:x.shape[1]] = np.copy(x)
    for layer in range(1, len(nn.layerNodeCounts)): # skip input layer
        for node in range(1, nn.layerNodeCounts[layer]): # skip augmented 1
            layerSum = np.sum(np.multiply(nn.nodes[layer-1,:], nn.weights[layer-1,node,:]))
            if layer == nn.layerCount: # output layer is linear combination
                nn.y = layerSum
            else: # hidden layers use sigmoid activation function
                nn.nodes[layer, node] = sigmoid(layerSum)

#####
# Author: Evan Hrouda
# Purpose: Implement the stochastic gradient descent neural network algorithm
#          using the square-loss function
#####
def NeuralNetwork_SGD(x, y, nn, GammaSchedule, T, checkConverge):
    from copy import deepcopy
    # weights are initialized during creation of NeuralNet nn
    idxs = np.arange(x.shape[0])
    gamma = GammaSchedule.gamma0
    iterations = 1
    
    lossList = []
    for epoch in range(T):
        # shuffle data by shuffling an array of indices
        np.random.shuffle(idxs)

        # for each training example
        for i in idxs:
            # calculate iteration gamma
            gamma = GammaSchedule.gamma0 / (1 + (GammaSchedule.gamma0 * iterations / GammaSchedule.d))

            # compute forward pass
            NeuralNetwork_Forwardpass(x[i], nn)
            # compute gradient of loss using backpropagation            
            NeuralNetwork_Backpropagation(y[i], nn)
            # update the weights using calculated gradient and treating this example as entire dataset
            nn.weights = np.subtract(nn.weights, x.shape[0]*gamma*nn.dweights)

            iterations += 1

        # if checkConverge is true, need to return a list of loss values
        if checkConverge == True:
            lossSum = 0
            for i in idxs:
                NeuralNetwork_Forwardpass(x[i], nn)
                lossSum += 0.5 * (nn.y - y[i])**2
            lossList.append(lossSum)

    return deepcopy(nn), lossList

#####
# Author: Evan Hrouda
# Purpose: predict the labels of the x matrix using the weight vector of the NeuralNet nn
#          with learned weights from the stochastic sub-gradient descent algorithm
#####
def NeuralNetwork_SGD_predict(x, nn):
    predictions = []
    for ex in x:
        NeuralNetwork_Forwardpass(ex, nn)
        p = nn.y
        if p < 0:
            predictions.append(-1)
        else:
            predictions.append(1)
    return np.array(predictions)
