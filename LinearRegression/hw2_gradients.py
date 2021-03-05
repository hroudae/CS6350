##########
# Author: Evan Hrouda
# Purpose: Perform the batch gradient expirements as described in homework 2
##########
import numpy as np
from numpy.linalg import inv

import sys
sys.path.append("../DecisionTree")
sys.path.append("../PreProcess")

import DecisionTree # for parse csv
import utilities
import BatchGradientDescent
import StochasticGradientDescent

print()
print()
print()
print()
print("********** Part 4a **********")
print("Batched gradient descent experiment")

train_data = "concrete/train.csv"
test_data = "concrete/test.csv"

cols = ['Cement', 'Slag', 'Fly ash', 'Water', 'SP', 'Coarse Aggr', 'Fine Aggr', 'slump']
examples_train = DecisionTree.parseCSV(train_data, cols)
examples_test = DecisionTree.parseCSV(test_data, cols)

r = 0.01
x,y = utilities.createInputMatrices(examples_train, 'slump')
wghts, costs = BatchGradientDescent.GradientDescent(x, y, r)

print(f"Learning rate: {r}")
print(f"The learned weight vector: {wghts}")

test_x, test_y = utilities.createInputMatrices(examples_test, 'slump')
print(f"Test data cost function value: {utilities.costValue(wghts,test_x,test_y)}")

# with open("concrete_costvals_bgd.csv", "w") as f:
#     f.write("Iteration,Cost Function Value\n")
#     for i in range(len(costs)):
#         f.write(f"{i+1},{costs[i]}\n")

# print("The cost function value at each iteration was written to concrete_costvals_bgd.csv")




print()
print()
print()
print()
print("********** Part 4b **********")
print("Stochastic Gradient Descent experiment")

r = 0.001

wghts, costs, converge = StochasticGradientDescent.StochasticGradientDescent(x, y, r, 10000)

print(f"Learning rate: {r}")
print(f"The learned weight vector: {wghts}")
print(f"Converges: {converge}")
print(f"Test data cost function value: {utilities.costValue(wghts,test_x,test_y)}")

# with open("concrete_costvals_sgd.csv", "w") as f:
#     f.write("Iteration,Cost Function Value\n")
#     for i in range(len(costs)):
#         f.write(f"{i+1},{costs[i]}\n")

# print("The cost function value at each iteration was written to concrete_costvals_sgd.csv")



print()
print()
print()
print()
print("********** Part 4c **********")
print("Find optimal weight vector with analytical form")

print("w = (XX^T)^-1 XY")
newx = []
for i in range(len(x[0])):
    row = []
    for j in range(len(x)):
        row.append(x[j][i])
    newx.append(row)

newx = np.array(newx)
xxt = np.matmul(newx, newx.T)
xxtinv = inv(xxt)
finalres = np.matmul(np.matmul(xxtinv,newx),y)
print(finalres)
print(f" Test data cost function value: {utilities.costValue(list(finalres),test_x,test_y)}")
