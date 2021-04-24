##########
# Author: Evan Hrouda
# Purpose: Implement stochastic gradient descent logisitc regression using
#          MAP and ML estimation
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
# Purpose: Sigmoid function
#####
def sigmoid(x):
    from math import exp
    # prevent overflow
    if x >= 700:
        return 1
    elif x <= -700:
        return 0
    return 1 / (1 + exp(-1*x))

#####
# Author: Evan Hrouda
# Purpose: Implement stochastic gradient descent logistic regression using MAP estimation
#####
def LogisticRegression_SGD_MAP(x, y, T, v, GammaSchedule, checkConverge):
    import math
    wghts = np.zeros((1,x.shape[1]))
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

            # compute gradient of loss function and update weight
            dL = np.subtract((1/v)*wghts, (1 - sigmoid(y[i]*np.dot(wghts, x[i].T))) * x.shape[0] * y[i] * x[i])
            wghts = np.subtract(wghts, gamma*dL)

            iterations += 1

        # if checkConverge is true, need to return a list of loss values
        if checkConverge == True:
            lossSum = (1 / (2*v)) * np.dot(wghts, wghts.T)
            for i in idxs:
                lossSum += math.log(1 + math.exp(-y[i] * np.dot(wghts, x[i].T)))
            lossList.append(np.asscalar(lossSum))

    return wghts, lossList

#####
# Author: Evan Hrouda
# Purpose: Predict labels of using the weight vector learned with 
#          stochastic gradient descent logistic regression using MAP estimation
#####
def LogisticRegression_SGD_MAP_predict(x, w):
    predictions = []
    for ex in x:
        p = np.dot(w, ex.T)
        if p < 0:
            predictions.append(-1)
        else:
            predictions.append(1)
    return np.array(predictions)

#####
# Author: Evan Hrouda
# Purpose: Implement stochastic gradient descent logistic regression using ML estimation
#####
def LogisticRegression_SGD_ML(x, y, T, GammaSchedule, checkConverge):
    import math
    wghts = np.zeros((1,x.shape[1]))
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

            # compute gradient of loss function and update weight
            dL = -(1 - sigmoid(y[i]*np.dot(wghts, x[i].T))) * x.shape[0] * y[i] * x[i]
            wghts = np.subtract(wghts, gamma*dL)

            iterations += 1

        # if checkConverge is true, need to return a list of loss values
        if checkConverge == True:
            lossSum = 0
            for i in idxs:
                power = np.asscalar(-y[i] * np.dot(wghts, x[i].T))
                if power > 700: # overflow
                    lossSum += power
                    break
                lossSum += math.log(1 + math.exp(power))
            lossList.append(lossSum)

    return wghts, lossList

#####
# Author: Evan Hrouda
# Purpose: Predict labels of using the weight vector learned with 
#          stochastic gradient descent logistic regression using ML estimation
#####
def LogisticRegression_SGD_ML_predict(x, w):
    predictions = []
    for ex in x:
        p = np.dot(w, ex.T)
        if p < 0:
            predictions.append(-1)
        else:
            predictions.append(1)
    return np.array(predictions)
