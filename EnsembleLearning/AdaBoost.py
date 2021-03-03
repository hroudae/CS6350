##########
# Author: Evan Hrouda
# Purpose: Implement the AdaBoost algorithm using decision tree stumps
##########
import sys
sys.path.append("../DecisionTree")

import DecisionTree

#####
# Author: Evan Hrouda
# Purpose: Replace the binary labal with +1 or -1 for the AdaBoost algorithm
#####
def stringBinaryLabel2numerical(data, attrDict, labelCol, negVal, posVal):
    import copy
    newData = copy.deepcopy(data)
    newAttrDict = copy.deepcopy(attrDict)
    for example in newData:
        if example[labelCol] == negVal:
            example[labelCol] = -1
        elif example[labelCol] == posVal:
            example[labelCol] = 1
    newAttrDict[labelCol] = [-1, 1]
    return newData, newAttrDict

#####
# Author: Evan Hrouda
# Purpose: Replace the numerical labels (+1 or -1) with the original string labels
#####
def numericalLabel2string(data, attrDict, labelCol, negVal, posVal):
    import copy
    newData = copy.deepcopy(data)
    newAttrDict = copy.deepcopy(attrDict)
    for example in newData:
        if example[labelCol] == -1:
            example[labelCol] = negVal
        elif example[labelCol] == 1:
            example[labelCol] = posVal
    newAttrDict[labelCol] = [negVal, posVal]
    return newData, newAttrDict

#####
# Author: Evan Hrouda
# Purpose: Implement the AdaBoost algorithm
#####
def AdaBoost(data, attrDict, labelCol, gainMethod, T):
    import math

    d_weights = [1/len(data) for i in range(len(data))] # the weights for each example

    a_list = []
    hyp_list = []

    # run for the specified number of iterations
    for i in range(0, T):
        # create a decision stump h
        root = DecisionTree.Tree(None)
        root.depth = 0
        DecisionTree.ID3(data, attrDict, labelCol, root, 1, gainMethod, d_weights)
        hyp_list.append(root)

        # compute vote a = 0.5*ln((1-e) / e)
        stumpPredict = DecisionTree.predict(data, 'prediction', root)
        
        e = 0.0 # sum of all weights where prediction is not the same as the label
        for idx, example in enumerate(data):
            if example[labelCol] != stumpPredict[idx]['prediction']:
                e += d_weights[idx]
        
        a = 0.5 * math.log2((1-e) / e)
        a_list.append(a)

        # update weights for the training example
        # D_i = (D_i/Z) * exp(-a * y_i * h(x_i))
        for idx, example in enumerate(data):
            d_weights[idx] = (d_weights[idx]) * math.exp(-a * example[labelCol] * stumpPredict[idx]['prediction'])

        Z = sum(d_weights)
        d_weights[:] = [d / Z for d in d_weights]

    # return final hyp sgn(sum over T (a*h(x_i)))
    return a_list, hyp_list

#####
# Author: Evan Hourda
# Purpose: using the specifided vote list and hypothesis list, predict the label of the data set.
#####
def predict(data, predictCol, a_list, hyp_list):
    import copy
    predcitData = copy.deepcopy(data)
    for example in predcitData:
        hyp_sum = 0
        for idx, hyp in enumerate(hyp_list):
            hyp_sum += a_list[idx]*DecisionTree.predict_example(example, predictCol, hyp)
        if hyp_sum < 0:
            example[predictCol] = -1
        else:
            example[predictCol] = 1
    return predcitData

#####
# Author: Evan Hourda
# Purpose: Predict the label of each example using the tree stumps in the hypothesis list.
#          Return the error for each stump. NOTE: the prediction is not returned, only the error
#####
def stumpErrors(data, labelCol, predictCol, hyp_list):
    import copy
    stumpErrorList = []

    for idx, hyp in enumerate(hyp_list):
        predictdata = DecisionTree.predict(data, predictCol, hyp)
        total = 0
        wrong = 0
        for example in predictdata:
            if example[labelCol] != example[predictCol]:
                wrong += 1
            total += 1
        stumpErrorList.append(wrong/total)
    return stumpErrorList
