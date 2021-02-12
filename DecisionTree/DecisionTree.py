##########
# Author: Evan Hrouda
# Purpose: Implement ID3 decision tree learning algorithm using three gain
#          methods: information gain, majority error, and gini index
##########
import csv
from enum import Enum


# Define the various information gain methods
class GainMethods(Enum):
    ENTROPY = 1
    MAJORITY = 2
    GINI = 3

# Tree class
class Tree:
    def __init__(self, val):
        self.children = []
        self.parent = None
        self.attrSplit = None
        self.attrValue = val
        self.label = None
        self.common = None
        self.depth = None


#####
# Author: Evan Hrouda
# Purpose: Parse a csv file with the attributes
#####
def parseCSV(csvFilePath, cols):
    data = []

    with open(csvFilePath, 'r') as csvFile:
        csvReader = csv.reader(csvFile, delimiter=',')

        for row in csvReader:
            thisDict = {}
            i = 0
            for attr in cols:
                thisDict[attr] = row[i]
                i += 1
            data.append(thisDict)

    return data

#####
# Author: Evan Hrouda
# Purpose: Check all examples to see if they have the same label
#####
def sameLabel(data, labelCol):
    lbl = data[0][labelCol]
    for example in data:
        if example[labelCol] != lbl:
            return False
    return True

#####
# Author: Evan Hrouda
# Purpose: Find the most common label in a group of data
#####
def common(data, attr, labelCol):
    countDict = {}
    for lbl in attr[labelCol]:
        countDict[lbl] = 0
    
    for example in data:
        countDict[example[labelCol]] = countDict[example[labelCol]] + 1

    vals = list(countDict.values())
    keys = list(countDict.keys())
    return keys[vals.index(max(vals))]

#####
# Author: Evan Hrouda
# Purpose: return a list of all the examples with the specified value
#          for the given attribute
#####
def splitData(data, attr, v):
    newData = []
    for example in data:
        if example[attr] == v:
            newData.append(example)

    return newData

#####
# Author: Evan Hrouda
# Purpose: find the purity of the data with the given attribute value
#          using the specified method
#####
def purity(val, attr, data, labelList, labelCol, gainMethod):
    import math
    # create a dictionary of the labels to count their occurence
    labelCount = {}
    for lbl in labelList:
        labelCount[lbl] = 0

    # count the occurence of each label value in the given data if they have the
    # specified attribute value
    totalOccurences = 0
    for example in data:
        if val == "all": # for finding the purity of the entire data subset
            labelCount[example[labelCol]] = labelCount[example[labelCol]] + 1
            totalOccurences += 1
        elif example[attr] == val:
            labelCount[example[labelCol]] = labelCount[example[labelCol]] + 1
            totalOccurences += 1

    # calculate the purity of the data using the specified method
    if gainMethod == GainMethods.ENTROPY:
        entropy = 0
        for lbl in labelList:
            if totalOccurences > 0:
                p_i = labelCount[lbl]/totalOccurences
                if p_i > 0:
                    entropy += (p_i * math.log2(p_i))
        return (-1*entropy), totalOccurences
    elif gainMethod == GainMethods.GINI:
        giniSum = 0
        for lbl in labelList:
            if totalOccurences > 0:
                p_i = labelCount[lbl]/totalOccurences
                giniSum += (p_i * p_i)
        return (1-giniSum), totalOccurences
    elif gainMethod == GainMethods.MAJORITY:
        vals = list(labelCount.values())
        keys = list(labelCount.keys())
        maxIndx = vals.index(max(vals))
        errorSum = 0
        for i in range(0,len(vals)):
            if i != maxIndx and totalOccurences > 0:
                errorSum += vals[i]
        if totalOccurences > 0:
            return (errorSum/totalOccurences), totalOccurences
        else:
            return 0, totalOccurences

#####
# Author: Evan Hrouda
# Purpose: find the attribute with the highest information gain to split on using the
#          specified information gain technique
#####
def best(data, attrList, labelCol, gainMethod):
    # calculate the entropy/ME/gini of full set
    setPurity, totalCount = purity("all", None, data, attrList[labelCol], labelCol, gainMethod)

    # calc the entropy/ME/gini of each attribute
    attrGains = {}
    for attr in attrList:
        if attr == labelCol:
            continue
        attrPuritySum = 0
        for val in attrList[attr]: # calc the entropy/ME/gini of ea val of the att and expexted entropy/ME/gini of the attr
            attrValPurity, occur= purity(val, attr, data, attrList[labelCol], labelCol, gainMethod)
            attrPuritySum += (occur/totalCount) * attrValPurity

        # calc the gain of the attr
        attrGains[attr] = setPurity - attrPuritySum
    
    # return attr w/ max info gain
    vals = list(attrGains.values())
    keys = list(attrGains.keys())
    return keys[vals.index(max(vals))]


#####
# Author: Evan Hrouda
# Purpose: Perform the ID3 algorithm
#####
def ID3(data, hdrs, attr, labelCol, node, maxDepth, gainMethod):
    import copy
    if not attr: # If attributes is empty, return leaf node with most common label
        node.label = node.parent.common
        return
    if sameLabel(data, labelCol): # if all examples have the same label, this branch is done
        node.label = data[0][labelCol]
        return

    node.common = common(data, attr, labelCol) # find most common label in data

    # if the max tree depth has been reached, just label everything with the most common
    if node.depth == maxDepth:
        node.label = node.common
        return

    node.attrSplit = best(data, attr, labelCol, gainMethod) # TODO

    for v in attr[node.attrSplit]:
        if v == labelCol:
            continue
        # create a new tree node
        child = Tree(v)
        child.parent = node
        child.depth = child.parent.depth + 1
        node.children.append(child)

        # split the data over the attribute value
        dataSplit = splitData(data, node.attrSplit, v) # TODO

        if not dataSplit: # TODO
            child.label = child.parent.common
        else:
            newAttrList = copy.deepcopy(attr)
            del newAttrList[node.attrSplit] # delete the attribute we just split on
            ID3(dataSplit, hdrs, newAttrList, labelCol, child, maxDepth, gainMethod)

    return node

#####
# Author: Evan Hrouda
# Purpose: using the specifided decision tree, predict the label of the data set.
#####
def predict(data, attrDict, predictCol, root):
    import copy
    for example in data:
        prediction = None
        node = root#copy.deepcopy(root)
        while node.label == None:
            for child in node.children:
                if child.attrValue == example[node.attrSplit]:
                    node = child
                    break
        example[predictCol] = node.label
    return data
