##########
# Author: Evan Hrouda
# Purpose: Run the experiments described in the Homework 5 handout
##########
import LogisticRegression

import numpy as np
from matplotlib import pyplot as plt

train_data = "bank-note/train.csv"
test_data = "bank-note/test.csv"

print()
print()
print("********** Part 2a **********")
print("Logistic regression with stochastic gradient descent using MAP estimation")

x, y = LogisticRegression.parseCSV(train_data, True)
x_test, y_test = LogisticRegression.parseCSV(test_data, True)

checkConverge = False
T = 100

gamma0 = 0.008
d = 0.75
g = LogisticRegression.GammaSchedule(gamma0, d)
vList = [0.01, 0.1, 0.5, 1, 3, 5, 10, 100]

print(f"With gamma0 = {gamma0} and d = {d}")
print("v\tTrain Error\tTest Error")
for i, v in enumerate(vList):
    w, loss = LogisticRegression.LogisticRegression_SGD_MAP(x, y, T, v, g, checkConverge)

    train_predicts = LogisticRegression.LogisticRegression_SGD_MAP_predict(x, w)
    numWrong = sum(abs(train_predicts-y) / 2)
    train_err = numWrong/len(y)

    test_predicts = LogisticRegression.LogisticRegression_SGD_MAP_predict(x_test, w)
    numWrong = sum(abs(test_predicts-y_test) / 2)
    test_err = numWrong/len(y_test)

    print(f"{v}\t{train_err:.7f}\t{test_err:.7f}")
    
    # plots to check for convergence
    if checkConverge:
        plt.title("Objective Function Value vs Epochs")
        plt.xlabel("Iterations")
        plt.ylabel("Objective Function Value")
        plt.plot([k for k in range(1,len(loss)+1)], loss)
        plt.savefig(f"2a_{v}.png", bbox_inches='tight')
        print(f"Convergence plot save as: 2a_{v}.png")
        plt.close()




print()
print()
print("********** Part 2b **********")
print("Logistic regression with stochastic gradient descent using ML estimation")

checkConverge = False
T = 100

gamma0 = 0.01
d = 0.8
g = LogisticRegression.GammaSchedule(gamma0, d)

print(f"With gamma0 = {gamma0} and d = {d}")
print("Train Error\tTest Error")
w, loss = LogisticRegression.LogisticRegression_SGD_ML(x, y, T, g, checkConverge)

train_predicts = LogisticRegression.LogisticRegression_SGD_ML_predict(x, w)
numWrong = sum(abs(train_predicts-y) / 2)
train_err = numWrong/len(y)

test_predicts = LogisticRegression.LogisticRegression_SGD_ML_predict(x_test, w)
numWrong = sum(abs(test_predicts-y_test) / 2)
test_err = numWrong/len(y_test)

print(f"{train_err:.7f}\t{test_err:.7f}")
    
# plots to check for convergence
if checkConverge:
    plt.title("Objective Function Value vs Epochs")
    plt.xlabel("Iterations")
    plt.ylabel("Objective Function Value")
    plt.plot([k for k in range(1,len(loss)+1)], loss)
    plt.savefig(f"2b.png", bbox_inches='tight')
    print(f"Convergence plot save as: 2b.png")
    plt.close()
