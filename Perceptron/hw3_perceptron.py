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

test_predictions = Perceptron.predict_StandardPerceptron(x_test, w)
# if prediction is different, difference will be +-2, if same, will be 0
numWrong = sum(abs(test_predictions-y_test) / 2)
print(f"Test Error after {T} epochs: {numWrong/len(y_test)}")



print()
print("********** Part 2b **********")
print("Voted Perceptron experiments")

r = 0.1
T = 10
wghts = Perceptron.VotedPerceptron(x, y, r, T)
# write weight vectors to file
with open("votedperceptron_weights.csv", 'w') as f:
    f.write(f"Weight Vector,Count\n")
    for wc in wghts:
        f.write(f"{wc[0]},{wc[1]}\n")
print("Weight vectors and their counts have been written to votedperceptron_weights.csv")

test_predictions = Perceptron.predict_VotedPerceptron(x_test, wghts)
# if prediction is different, difference will be +-2, if same, will be 0
numWrong = sum(abs(test_predictions-y_test) / 2)
print(f"Test Error after {T} epochs: {numWrong/len(y_test)}")




print()
print("********** Part 2c **********")
print("Average Perceptron experiments")

r = 0.1
T = 10
a = Perceptron.AveragedPerceptron(x, y, r, T)
print(f"Learned weight vector: {a}")

test_predictions = Perceptron.predict_AveragedPerceptron(x_test, a)
# if prediction is different, difference will be +-2, if same, will be 0
numWrong = sum(abs(test_predictions-y_test) / 2)
print(f"Test Error after {T} epochs: {numWrong/len(y_test)}")
