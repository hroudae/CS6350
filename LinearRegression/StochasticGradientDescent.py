##########
# Author: Evan Hrouda
# Purpose: Implement the Stochastic Gradient Descent algorithm for least mean square
##########
import utilities

#####
# Author: Evan Hrouda
# Purpose: Perform the stochastic gradient descent learning algorithm
#####
def StochasticGradientDescent(x, y, r, iterations):
    import random

    wghts = [0 for i in range(len(x[0]))]
    costs = [utilities.costValue(wghts, x, y)]
    converge = False

    for i in range(iterations):
        # randomly sample an example
        index = random.randrange(len(x))
        # update weight vector with the stochastic grad
        newWghts = []
        for j in range(len(wghts)):
            newWghts.append(wghts[j] + r*x[index][j]*(y[index] - utilities.dot(wghts,x[index])))
        wghts = newWghts
        # check convergence (calculate cost function)
        costVal = utilities.costValue(wghts, x, y)
        if abs(costVal - costs[-1]) > 10e-10:
            converge = True
        costs.append(costVal)

    return wghts, costs, converge
