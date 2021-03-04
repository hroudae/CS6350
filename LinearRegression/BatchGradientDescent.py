##########
# Author: Evan Hrouda
# Purpose: Implement the Batch Gradient Descent algorithm for least mean square
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
# Purpose: Perform the batach gradient descent learning algorithm
#####
def GradientDescent(x, y, r):
    import math
    wghts = [0 for i in range(len(x[0]))]
    costs = []

    norm = math.inf
    while norm > 10e-10:
        grad = []
        for j in range(len(wghts)):
            gradSum = 0
            for i in range(len(y)):
                gradSum += (x[i][j] * (y[i]-dot(wghts,x[i])))
            grad.append(-1*gradSum)

        newWghts = []
        for i in range(len(wghts)):
            newWghts.append(wghts[i] - (r*grad[i]))
        
        norm = math.sqrt(sum([(w-wm1)*(w-wm1) for w, wm1 in zip(newWghts, wghts)]))
        costs.append(costValue(wghts,x,y))
        wghts = newWghts
    costs.append(costValue(wghts,x,y))
    return wghts, costs

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
