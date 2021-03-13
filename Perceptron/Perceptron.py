##########
# Author: Evan Hrouda
# Purpose: Implement the various Perceptron learning algorithms
##########
import csv
import numpy as np

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
            thisExample = [1.0]
            thisExample += [float(row[i]) for i in range(len(row)-1)]
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
# Purpose: implement the standard Perceptron algorithm
#####
def StandardPerceptron(x, y, r, T):
    wghts = np.zeros((1,x.shape[1]))
    idxs = np.arange(x.shape[0])

    for epoch in range(T):
        # shuffle data by shuffling an array of indices
        np.random.shuffle(idxs)

        # for each training example
        for i in idxs:
            if (y[i]*np.dot(wghts,x[i].T)) <= 0:
                wghts = wghts + r*y[i]*x[i]

    return wghts

#####
# Author: Evan Hrouda
# Purpose: predict the labels of the x matrix using the weight vector
#          from the standard perceptron algorithm
#####
def predict_StandardPerceptron(x, w):
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
# Purpose: Implementation of the Voted Perceptron algorithm
#####
def VotedPerceptron(x, y, r, T):
    wghts = np.zeros((1,x.shape[1]))
    idxs = np.arange(x.shape[0])

    wght_list = []
    cnt = 0
    m = 0

    for epoch in range(T):
        # shuffle data by shuffling an array of indices
        np.random.shuffle(idxs)

        # for each training example
        for i in idxs:
            if (y[i]*np.dot(wghts,x[i].T)) <= 0:
                if m > 0:
                    wght_list.append((wghts, cnt))
                wghts = wghts + r*y[i]*x[i]
                m += 1
                cnt = 1
            else:
                cnt += 1

    return wght_list

#####
# Author: Evan Hrouda
# Purpose: Predict the labels of the x matrix using the weight vector list
#          from the voted perceptron algorithm
#####
def predict_VotedPerceptron(x, wght_list):
    predictions = []
    for ex in x:
        ex_sum = 0
        for w in wght_list:
            sgn = np.dot(w[0], ex.T)
            if sgn < 0:
                ex_sum -= w[1]
            else:
                ex_sum += w[1]
        if ex_sum < 0:
            predictions.append(-1)
        else:
            predictions.append(1)
    return np.array(predictions)

#####
# Author: Evan Hrouda
# Purpose: Implementation of the Averaged Perceptron algorithm
#####
def AveragedPerceptron(x, y, r, T):
    wghts = np.zeros((1,x.shape[1]))
    a = np.zeros((1,x.shape[1]))
    idxs = np.arange(x.shape[0])

    for epoch in range(T):
        # shuffle data by shuffling an array of indices
        np.random.shuffle(idxs)

        # for each training example
        for i in idxs:
            if (y[i]*np.dot(wghts,x[i].T)) <= 0:
                wghts = wghts + r*y[i]*x[i]
            a += wghts

    return a


#####
# Author: Evan Hrouda
# Purpose: Predict the labels of the x matrix using the weight vector
#          from the averaged perceptron algorithm
#####
def predict_AveragedPerceptron(x, a):
    predictions = []
    for ex in x:
        p = np.dot(a, ex.T)
        if p < 0:
            predictions.append(-1)
        else:
            predictions.append(1)
    return np.array(predictions)
