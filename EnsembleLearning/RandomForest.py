##########
# Author: Evan Hrouda
# Purpose: Implement the Random Forests machine learning algorithm
##########
from math import inf
import sys
sys.path.append("../DecisionTree")

import DecisionTree

#####
# Author: Evan Hrouda
# Purpose: Learn a Random Forests. Return a list of decision trees to use for prediction
#####
def RandomForests(data, attrDict, labelCol, gainMethod, T, featureSetSize, samplesize):
    import random, math
    forest = []

    for i in range(T):
        # draw m samples uniformly random with replacement from data
        samples = random.choices(data, k=math.ceil(samplesize*len(data)))

        # learn a full decision tree based on those samples with no depth limit
        root = DecisionTree.Tree(None)
        root.depth = 0
        DecisionTree.ID3_RandTree(samples, attrDict, labelCol, root, inf, gainMethod, None, featureSetSize)
        forest.append(root)

    # return the list of trees
    return forest

#####
# Author: Evan Hrouda
# Purpose: Using the list of trees from RandomForests, predict
#          the label of the data using a plurality vote
#####
def predict(data, predictCol, forest):
    import copy
    predcitData = copy.deepcopy(data)

    for example in predcitData:
        # pick the label predicted the most
        example[predictCol] = predict_example(example, predictCol, forest)

    return predcitData

#####
# Author: Evan Hrouda
# Purpose: Predict a single example using plurality vote
#####
def predict_example(example, predictCol, forest):
    labelVotes = {}
    # predict the example label using all the trees
    for root in forest:
        thisPredict = DecisionTree.predict_example(example, predictCol, root)
        if thisPredict not in labelVotes:
            labelVotes[thisPredict] = 1
        else:
            labelVotes[thisPredict] += 1
    # pick the label predicted the most
    return max(labelVotes, key=labelVotes.get)
