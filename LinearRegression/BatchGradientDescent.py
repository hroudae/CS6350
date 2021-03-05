##########
# Author: Evan Hrouda
# Purpose: Implement the Batch Gradient Descent algorithm for least mean square
##########
import utilities

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
                gradSum += (x[i][j] * (y[i]-utilities.dot(wghts,x[i])))
            grad.append(-1*gradSum)

        newWghts = []
        for i in range(len(wghts)):
            newWghts.append(wghts[i] - (r*grad[i]))
        
        norm = math.sqrt(sum([(w-wm1)*(w-wm1) for w, wm1 in zip(newWghts, wghts)]))
        costs.append(utilities.costValue(wghts,x,y))
        wghts = newWghts
    costs.append(utilities.costValue(wghts,x,y))
    return wghts, costs
