##########
# Author: Evan Hrouda
# Purpose: Run the experiments described in the Homework 5 handout
##########
import NeuralNetwork

import numpy as np
from matplotlib import pyplot as plt

train_data = "bank-note/train.csv"
test_data = "bank-note/test.csv"

print()
print("********** Part 3a **********")
print("Backpropagation test with values from paper problem 3.")
nn = NeuralNetwork.NeuralNet(3, 3, [3, 3], False)
w = [
        [ # layer 0->1, w^1s
            [ 0,  0,  0], # empty
            [-1, -2, -3], # w01, w11, w21
            [ 1,  2,  3]  # w02, w12, w22
        ],
        [ # layer 1->2, w^21
            [ 0,  0,  0], # empty
            [-1, -2, -3], # w01, w11, w21
            [ 1,  2,  3]  # w02, w12, w22
        ],
        [ # layer 2->3, w^3s (output)
            [0, 0, 0], # empty
            [-1, 2, -1.5], # w01, w11, w21
            [0, 0, 0]  # empty
        ]
    ]
nn.weights = np.array(w)

n = [
        [1, 1, 1], # input
        [1, 0.00247, 0.99753], # hidden layer 1
        [1, 0.018, 0.982] # hidden layer 2
    ]
nn.nodes = np.array(n)

nn.y = -2.437

print(f"layers: {nn.layerCount}")
print(f"nodes: {nn.nodes}")
print(f"w: {nn.weights}")
print(f"y: {nn.y}")

NeuralNetwork.NeuralNetwork_Backpropagation(1, nn)
print(f"dw: {nn.dweights}")

print()
print("Forward Pass test with values from paper problem 3.")
NeuralNetwork.NeuralNetwork_Forwardpass(np.matrix([1,1,1]), nn)
print(f"nodes: {nn.nodes}")
print(f"y: {nn.y}")





print()
print()
print("********** Part 3b **********")
print("Neural Network SGD with bank data and intializing the weights with random values from the Gaussian distribution")

x, y = NeuralNetwork.parseCSV(train_data, True)
x_test, y_test = NeuralNetwork.parseCSV(test_data, True)

checkConverge = False
T = 100

gammaList = [                 # gamma0, d
    NeuralNetwork.GammaSchedule(1/8720, 40), # 0.15, 30 - 0.1, 35
    NeuralNetwork.GammaSchedule(1/17440, 25), # 10 0.1, 15 - 0.05, 20
    NeuralNetwork.GammaSchedule(1/34880, 35), # 25 0.1, 20 - 0.05, 25 - 0.05, 30
    NeuralNetwork.GammaSchedule(7/87200, 25), # 50 0.075, 17.5 - 0.07, 18
    NeuralNetwork.GammaSchedule(1/87200, 10)  # 100 0.02, 2 - 0.01, 2.5
    ]

widths = [5, 10, 25, 50, 100]

print("Width\tTrain Error\tTest Error\tgamma0\t\td")
for i, width in enumerate(widths):
    nn = NeuralNetwork.NeuralNet(3, x.shape[1], [width, width], True)
    nn_learned, loss = NeuralNetwork.NeuralNetwork_SGD(x, y, nn, gammaList[i], T, checkConverge)

    train_predicts = NeuralNetwork.NeuralNetwork_SGD_predict(x, nn_learned)
    numWrong = sum(abs(train_predicts-y) / 2)
    train_err = numWrong/len(y)

    test_predicts = NeuralNetwork.NeuralNetwork_SGD_predict(x_test, nn_learned)
    numWrong = sum(abs(test_predicts-y_test) / 2)
    test_err = numWrong/len(y_test)

    print(f"{width}\t{train_err:.7f}\t{test_err:.7f}\t{gammaList[i].gamma0:.7f}\t{gammaList[i].d}")
    
    # plots to check for convergence
    if checkConverge:
        plt.title("Squared Loss vs Epochs")
        plt.xlabel("Iterations")
        plt.ylabel("Squared Loss")
        plt.plot([k for k in range(1,len(loss)+1)], loss)
        plt.savefig(f"3b_{width}.png", bbox_inches='tight')
        print(f"Convergence plot save as: 3b_{width}.png")
        plt.close()


print()
print()
print("********** Part 3c **********")
print("Neural Network SGD with bank data and intializing the weights to 0")

print("Width\tTrain Error\tTest Error\tgamma0\t\td")
for i, width in enumerate(widths):
    nn = NeuralNetwork.NeuralNet(3, x.shape[1], [width, width], False)
    nn_learned, loss = NeuralNetwork.NeuralNetwork_SGD(x, y, nn, gammaList[i], T, False)

    train_predicts = NeuralNetwork.NeuralNetwork_SGD_predict(x, nn_learned)
    numWrong = sum(abs(train_predicts-y) / 2)
    train_err = numWrong/len(y)

    test_predicts = NeuralNetwork.NeuralNetwork_SGD_predict(x_test, nn_learned)
    numWrong = sum(abs(test_predicts-y_test) / 2)
    test_err = numWrong/len(y_test)

    print(f"{width}\t{train_err:.7f}\t{test_err:.7f}\t{gammaList[i].gamma0:.7f}\t{gammaList[i].d}")
