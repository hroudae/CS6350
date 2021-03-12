###########
# Author: Evan Hrouda
# Purpose: Implement the Bagged Decision Trees algorithm
###########
from math import inf
import sys
sys.path.append("../DecisionTree")

import DecisionTree


#####
# Author: Evan Hrouda
# Purpose: Perform the bagged decision tree algorithm
#####
def BaggedDecisionTrees(data, attrDict, labelCol, gainMethod, T, samplesize):
    import random, math
    trees = []

    for i in range(T):
        # draw m samples uniformly random with replacement from data
        samples = random.choices(data, k=math.ceil(samplesize*len(data)))

        # learn a decision tree based on those samples with no depth limit
        root = DecisionTree.Tree(None)
        root.depth = 0
        DecisionTree.ID3(samples, attrDict, labelCol, root, inf, gainMethod, None)
        trees.append(root)

    # return the list of trees
    return trees

#####
# Author: Evan Hrouda
# Purpose: Using the list of trees from the BaggedDecisionTrees and predict
#          the label of the data. Uses a plurality vote to predict
#####
def predict(data, predictCol, trees):
    import copy
    predictData = copy.deepcopy(data)

    for example in predictData:
        # pick the label predicted the most
        example[predictCol] = predict_example(example, predictCol, trees)

    return predictData

#####
# Author: Evan Hrouda
# Purpose: Predict a single example using plurality vote
#####
def predict_example(example, predictCol, trees):
    labelVotes = {}
    # predict the example label using all the trees
    for root in trees:
        thisPredict = DecisionTree.predict_example(example, predictCol, root)
        if thisPredict not in labelVotes:
            labelVotes[thisPredict] = 1
        else:
            labelVotes[thisPredict] += 1
    # pick the label predicted the most
    return max(labelVotes, key=labelVotes.get)
