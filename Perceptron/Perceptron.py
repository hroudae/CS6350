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
#####
def predict_StandardPerceptron(x, w):
    predictions = []
    for i in x:
        p = np.dot(w, i.T)
        if p < 0:
            predictions.append(-1)
        else:
            predictions.append(1)
    return np.array(predictions)


