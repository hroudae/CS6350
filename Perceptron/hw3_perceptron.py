##########
# Author: Evan Hrouda
# Purpose: Run the experiments described in the Homework 3 handout
##########
import numpy as np

import Perceptron

train_data = "bank-note/train.csv"
test_data = "bank-note/test.csv"


print("********** Part 2a **********")
print("Standard Perceptron experiments")

x, y = Perceptron.parseCSV(train_data, True)
x_test, y_test = Perceptron.parseCSV(test_data, True)

r = 0.1
T = 10
w = Perceptron.StandardPerceptron(x, y, r, T)
print(f"Learned weight vector: {w}")
train_predictions = Perceptron.predict_StandardPerceptron(x, w)

test_predictions = Perceptron.predict_StandardPerceptron(x_test, w)
# if prediction is different, difference will be +-2, if same, will be 0
numWrong = sum(abs(test_predictions-y_test) / 2)
print(f"Test Error after {T} epochs: {numWrong/len(y_test)}")




print()
print()
print()
print("********** Part 2a **********")
print("Voted Perceptron experiments")
