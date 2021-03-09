##########
# Author: Evan Hrouda
# Purpose: Various utility functions for linear regression
##########

#####
# Author: Evan Hrouda
# Purpose: Create a matrix x of the examples with a leading 1 for the bias term
#          and a vector y for the labels
#####
def createInputMatrices(data, labelCol):
    x = []
    y = []
    for example in data:
        exampleList = [1] # one in front for the bias term
        for attr in example:
            if attr == labelCol:
                y.append(float(example[attr]))
            else:
                exampleList.append(float(example[attr]))
        x.append(exampleList)
    return x, y

#####
# Author: Evan Hrouda
# Purpose: Find the dot product between w and x
#####
def dot(w, x):
    return sum([wk*xk for wk,xk in zip(w,x)])

#####
# Author: Evan Hrouda
# Purpose: Calculated the cost function value for the given vector w, x, and y
#####
def costValue(w,x,y):
    costSum = 0
    for i in range(len(y)):
        costSum += (y[i] - dot(w,x[i]))**2
    return 0.5*costSum

