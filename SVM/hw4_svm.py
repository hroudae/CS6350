##########
# Author: Evan Hrouda
# Purpose: Run the experiments described in the Homework 4 handout
##########
import numpy as np
from matplotlib import pyplot as plt

import SVM

train_data = "bank-note/train.csv"
test_data = "bank-note/test.csv"


print("********** Part 2 **********")
print("SVM in primal domain with stochastic sub-gradient descent experiments")

x, y = SVM.parseCSV(train_data, True)
x_test, y_test = SVM.parseCSV(test_data, True)

T = 100
C = [100/873, 500/873, 700/873]

print()
print("Part 2a")
print("gamma = gamma_0 / (1 + t*(gamma_0)/d)")

gamma0 = 2.1
d = 1
gamma = SVM.GammaSchedule(1, gamma0, d)

plotCon = False # true if cost function should be plotted to check for convergence

print(f"With gamma0 = {gamma0} and d = {d}:")
print("C (*873)\tTrain Error\tTest Error\tW")

for i in range(len(C)):
    w, j = SVM.SVM_primalSGD(x, y, gamma, C[i], T, plotCon)
    train_predicts = SVM.predict_SVM_primalSGD(x, w)
    numWrong = sum(abs(train_predicts-y) / 2)
    train_err = numWrong/len(y)

    test_predicts = SVM.predict_SVM_primalSGD(x_test, w)
    numWrong = sum(abs(test_predicts-y_test) / 2)
    test_err = numWrong/len(y_test)
    
    print(f"{C[i]*873:.0f}\t\t{train_err:.7f}\t{test_err:.7f}\t{w}")

    # plots to check for convergence
    if plotCon:
        plt.title("J(w) vs epochs")
        plt.xlabel("Iterations")
        plt.ylabel("J(w)")
        plt.plot([k for k in range(1,len(j)+1)], j)
        plt.savefig(f"2a_{i}_d{d}_g{gamma0}.png", bbox_inches='tight')
    

print()
print("part 2b")
print("gamma = gamma_0 / (1 + t)")

gamma0 = 2.2
gamma = SVM.GammaSchedule(2, gamma0, 0)

plotCon = False

print(f"With gamma0 = {gamma0}")
print("C (*873)\tTrain Error\tTest Error\tW")

for i in range(len(C)):
    w, j = SVM.SVM_primalSGD(x, y, gamma, C[i], T, plotCon)
    train_predicts = SVM.predict_SVM_primalSGD(x, w)
    numWrong = sum(abs(train_predicts-y) / 2)
    train_err = numWrong/len(y)

    test_predicts = SVM.predict_SVM_primalSGD(x_test, w)
    numWrong = sum(abs(test_predicts-y_test) / 2)
    test_err = numWrong/len(y_test)
    
    print(f"{C[i]*873:.0f}\t\t{train_err:.7f}\t{test_err:.7f}\t{w}")

    # plots to check for convergence
    if plotCon:
        plt.title("J(w) vs epochs")
        plt.xlabel("Iterations")
        plt.ylabel("J(w)")
        plt.plot([k for k in range(1,len(j)+1)], j)
        plt.savefig(f"2b_{i}_g{gamma0}.png", bbox_inches='tight')



print()
print()
print()
print("********** Part 3 ************")
print("Part 3a")
print("Dual domain SVM")

print("C (*873)\tTrain Error\tTest Error\tW, b")
for i in range(len(C)):
    w, b = SVM.SVM_dual(x, y, C[i])
    train_predicts = SVM.predict_SVM_dual(x, w, b)
    numWrong = sum(abs(train_predicts-y) / 2)
    train_err = numWrong/len(y)

    test_predicts = SVM.predict_SVM_dual(x_test, w, b)
    numWrong = sum(abs(test_predicts-y_test) / 2)
    test_err = numWrong/len(y_test)
    
    print(f"{C[i]*873:.0f}\t\t{train_err:.7f}\t{test_err:.7f}\t{w}, {b}")




print()
print()
print()
print("********** Part 3 ************")
print("Part 3b")
print("Dual domain SVM with Gaussian Kernel")

glist = [0.1, 0.5, 1, 5, 100]

supportList = []

print("g\tC (*873)\tTrain Error\tTest Error\tNumber of Support Vectors")
for g in glist:
    for i in range(len(C)):
        a = SVM.SVM_dualKernelGaussian(x, y, C[i], g)
        train_predicts = SVM.predict_SVM_dualKernelGaussian(x, a, x, y, g)
        numWrong = sum(abs(train_predicts-y) / 2)
        train_err = numWrong/len(y)

        test_predicts = SVM.predict_SVM_dualKernelGaussian(x_test, a, x, y, g)
        numWrong = sum(abs(test_predicts-y_test) / 2)
        test_err = numWrong/len(y_test)
    
        print(f"{g}\t{C[i]*873:.0f}\t\t{train_err:.7f}\t{test_err:.7f}\t{(a!=0).sum()}")

        if C[i] == 500/873:
            supportList.append(a!=0)

print()
print("Shared support vectors between values of gamma with C=500/873:")
for i in range(len(glist)-1):
    count = 0
    for j in range(len(supportList[i])):
        if supportList[i][j] == supportList[i+1][j] and supportList[i][j] == True:
            count +=1
    print(f"{glist[i]}, {glist[i+1]}: {count}")




print()
print()
print()
print("********** Part 3 ************")
print("Part 3d")
print("Kernel Perceptron with Gaussian Kernel")

glist = [0.1, 0.5, 1, 5, 100]

print("g\tTrain Error\tTest Error")
for g in glist:
    c = SVM.Perceptron_Kernel_Gaussian(x, y, g, T)
    train_predicts = SVM.predict_Perceptron_Kernel_Gaussian(x, c, x, y, g)
    numWrong = sum(abs(train_predicts-y) / 2)
    train_err = numWrong/len(y)

    test_predicts = SVM.predict_Perceptron_Kernel_Gaussian(x_test, c, x, y, g)
    numWrong = sum(abs(test_predicts-y_test) / 2)
    test_err = numWrong/len(y_test)
    
    print(f"{g}\t{train_err:.7f}\t{test_err:.7f}")
