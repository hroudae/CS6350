##########
# Author: Evan Hrouda
# Purpose: Implement ID3 decision tree learning algorithm using the dataset in the
#          car directory. Training data is in car/train.csv and test data is in
#          car/test.csv.
##########
import csv
from enum import Enum


# Define the various information gain methods
class InformationGainMethods(Enum):
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

    vals = list(countDict.valus())
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
# Purpose: find the attribute with the highest information gain to split on using the
#          specified information gain technique
#####
def best(data, attrList, labelCol, infoGainMethod):
    # TODO: calculate the entropy/ME/gini of full set

    # TODO: calc the entropy/ME/gini of each attribute
    for attr in attrList:
        for val in attr: # TODO: calc the entropy/ME/gini of ea val of the att
            print()
        # TODO: calc the expexted entropy/ME/gini of the attr
        # TODO: calc the info gain of the attr
    
    # TODO: return attr w/ max info gain
            


#####
# Author: Evan Hrouda
# Purpose: Perform the ID3 algorithm
#####
def ID3(data, hdrs, attr, labelCol, node, maxDepth, infoGainMethod):
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

    node.attrSplit = best(data, attr, labelCol, infoGainMethod) # TODO

    for v in attr[node.attrSplit]:
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
            del attr[node.attrSplit] # delete the attribute we just split on
            ID3(dataSplit, hdrs, attr, labelCol, child)

    return node

# TODO: add cmd line args to get info
if __name__=="__main__":
    train_data = "/home/u1302032/CS6350/DecisionTree/car/train.csv"
    test_data = "/home/u1302032/CS6350/DecisionTree/car/test.csv"

    # columns
    cols = ['buying', 'maint', 'doors', 'persons', 'lug_boot', 'safety', 'label']

    # attribute values
    attrDict = {}
    attrDict["buying"] = ["vhigh", "high", "med", "low"]
    attrDict['maint'] = ['vhigh', 'high', 'med', 'low']
    attrDict['doors'] = ['2', '3', '4', '5more']
    attrDict['persons'] = ['2', '4', 'more']
    attrDict['lug_boot'] = ['small', 'med', 'big']
    attrDict['safety'] = ['low', 'med', 'high']
    attrDict['label'] = ['unacc', 'acc', 'good', 'vgood']

    rdr = parseCSV(train_data, cols)

    root = Tree()
    root.depth = 0

    ID3(rdr, cols, attrDict, 'label', root, 6, InformationGainMethods.ENTROPY)