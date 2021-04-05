##########
# Author: Evan Hrouda
# Purpose: Implement the various SVM learning algorithms
##########
import csv
from dataclasses import dataclass
import numpy as np
from numpy.linalg import multi_dot
from scipy.optimize import minimize


#####
# Author: Evan Hrouda
# Purpose: Set which gamma schedule to use and it's data values
# Schedules: 1 for gamm0/(1+(gamm0*t/d)), 2 for gamma0/(1+t)
#####
@dataclass
class GammaSchedule:
    schedule: int
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
# Purpose: Parse a csv file into a feature matrix x. For data with no label
#####
def parseCSV_NoLabel(csvFilePath):
    x = []

    with open(csvFilePath, 'r') as csvFile:
        csvReader = csv.reader(csvFile, delimiter=',')

        for row in csvReader:
            thisExample = [float(row[i]) for i in range(len(row))]
            thisExample += [1.0]
            x.append(thisExample)

    x = np.matrix(x)
    return x

#####
# Author: Evan Hrouda
# Purpose: Implement SVM in primal domain with stochastic sub-gradient descent
#####
def SVM_primalSGD(x, y, GammaSchedule, C, T, retCostList):
    wghts = np.zeros((1,x.shape[1]))
    idxs = np.arange(x.shape[0])
    gamma = GammaSchedule.gamma0
    iterations = 1

    j_list = [] # for checking convergence
    
    for epoch in range(T):
        # shuffle data by shuffling an array of indices
        np.random.shuffle(idxs)

        # for each training example
        for i in idxs:
            # calculate iteration gamma
            if GammaSchedule.schedule == 1:
                gamma = GammaSchedule.gamma0 / (1 + (GammaSchedule.gamma0 * iterations / GammaSchedule.d))
            elif GammaSchedule.schedule == 2:
                gamma = GammaSchedule.gamma0 / (1 + iterations)

            w0 = np.copy(wghts)
            w0[:,-1] = 0
            if (y[i]*np.dot(wghts,x[i].T)) <= 1:
                wghts = wghts - gamma*(w0) + gamma*C*x.shape[0]*y[i]*x[i]
            else:
                wghts = wghts - gamma*w0

            iterations += 1
            
            if retCostList:
                # append the j(w) for this epoch
                j = (0.5 * np.dot(wghts[:,:-1], wghts[:,:-1].T))
        
                for i in idxs:
                    j += (C * max(0, 1 - (y[i]*np.dot(wghts,x[i].T))))
                j_list.append(np.asscalar(j))

    return wghts, j_list

#####
# Author: Evan Hrouda
# Purpose: predict the labels of the x matrix using the weight vector
#          from the SVM in primal domain with stochastic sub-gradient descent algorithm
#####
def predict_SVM_primalSGD(x, w):
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
# Purpose: The objective function for dual SVM to minimize
#####
def SVM_dualObj(a, x, y):
    objSum = y * a * x
    objSum = 0.5 * np.dot(objSum, objSum.T)
    objSum -= np.sum(a)
    return np.asscalar(objSum)

#####
# Author: Evan Hrouda
# Purpose: Recover the optimal b using recovered weight vector and alphas that are >0 and <C
#####
def SVM_dualRecoverB(a, x, y, w, C):
    count = 0
    b = 0
    # find all alpha's that are 0 < alpha[i] < C, calculated optimal b, and average it
    for i in range(a.shape[0]):
        if a[i] > 0 and a[i] < C:
            b += y[i] - np.dot(w, x[i].T)
            count += 1
    return b/count

#####
# Author: Evan Hrouda
# Purpose: Implement SVM in dual domain
#####
def SVM_dual(x, y, C):
    # get rid of augmented one
    x = np.delete(x, x.shape[1]-1, 1)
    alpha0 = np.zeros((1,x.shape[0]))
    bnds = [(0, C) for i in range(x.shape[0])]
    cons = {'type': 'eq', 'fun': lambda a: np.asscalar(np.dot(a,y.T))}
    res = minimize(SVM_dualObj, alpha0, args=(x, y), method='SLSQP', bounds=bnds, constraints=cons)

    w = y * res.x * x
    b = SVM_dualRecoverB(res.x, x, y, w, C)

    return w, b

#####
# Author: Evan Hrouda
# Purpose: predict the labels of the x matrix using the weight vector an bias
#          from the SVM in dual domain
#####
def predict_SVM_dual(x, w, b):
    # get rid of augmented one
    x = np.delete(x, x.shape[1]-1, 1)
    predictions = []
    for ex in x:
        p = np.dot(w, ex.T) + b
        if p < 0:
            predictions.append(-1)
        else:
            predictions.append(1)
    return np.array(predictions)