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
def BaggedDecisionTrees(data, attrDict, labelCol, gainMethod, T):
    import random
    trees = []

    for i in range(T):
        # draw m samples uniformly random with replacement from data
        samples = random.choices(data, k=len(data))

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
    predcitData = copy.deepcopy(data)

    for example in predcitData:
        labelVotes = {}
        # predict the example label using all the trees
        for t in trees:
            thisPredict = DecisionTree.predict_example(example, predictCol, t)
            if thisPredict not in labelVotes:
                labelVotes[thisPredict] = 1
            else:
                labelVotes[thisPredict] += 1
        # pick the label predicted the mosst
        example[predictCol] = max(labelVotes, key=labelVotes.get)

    return predcitData
