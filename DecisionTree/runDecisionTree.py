##########
# Author: Evan Hrouda
# Purpose: Run the ID3 decision tree learning algorithm with the data specified
#          in the homework 1 handout
##########
from DecisionTree import *

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

# train_data = "/home/u1302032/CS6350/DecisionTree/tennis.csv"

# cols = ['outlook','temperature','humidity','wind','label']
# attrDict = {}
# attrDict['outlook'] = ['S','O','R']
# attrDict['temperature'] = ['H','M','C']
# attrDict['humidity'] = ['H','N','L']
# attrDict['wind'] = ['S','W']
# attrDict['label'] = ['+','-']

rdr = parseCSV(train_data, cols)
rdr_test = parseCSV(test_data, cols)

root = Tree(None)
root.depth = 0

ID3(rdr, cols, attrDict, 'label', root, 6, GainMethods.ENTROPY)

# print(root.attrSplit)
# print(root.children[0].attrValue + "\t" + root.children[1].attrValue + "\t" + root.children[2].attrValue)
# print(root.children[0].common + "\t" + "same"+ "\t" + root.children[2].common)
# print(root.children[0].attrSplit + "\t" + root.children[1].label + "\t" + root.children[2].attrSplit)
# print(root.children[0].children[0].attrValue + "\t" + root.children[0].children[1].attrValue + "\t" + root.children[0].children[2].attrValue + "\t" + "\t" + "\t" + root.children[2].children[0].attrValue + "\t" + root.children[2].children[1].attrValue)
# print(root.children[0].children[0].label + "\t" + root.children[0].children[1].label + "\t" + root.children[0].children[2].label + "\t" + "\t" + "\t" + root.children[2].children[0].label + "\t" + root.children[2].children[1].label)

#predictdata = predict(rdr_test, attrDict, "prediction", root)
predictdata = predict(rdr, attrDict, "prediction", root)

total = 0
wrong = 0
#print("actual\tprediction")
for example in predictdata:
    #print(example["label"] + "\t" + example["prediction"])
    if example["label"] != example["prediction"]:
        wrong += 1
    total += 1

print(str(wrong) + " / " + str(total) + " = " + str(wrong/total))