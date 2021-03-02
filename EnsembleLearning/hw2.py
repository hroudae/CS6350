##########
# Author: Evan Hrouda
# Purpose: Perform the required experiments as described in Homework 2 handout
##########
import sys
sys.path.append("../DecisionTree")
sys.path.append("../PreProcess")

import DecisionTree
import PreProcess
import AdaBoost
import BaggedTrees

# print("********** Part 2a **********")
# print("AdaBoost Experiment")

# # the training and test datasets
# train_data = "bank/train.csv"
# test_data = "bank/test.csv"

# maxDepth = 500

# # column names
# cols = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing',
#         'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays',
#         'previous', 'poutcome', 'y']

# # attribute values
# attrDict = {}
# attrDict["age"] = []
# attrDict['job'] = ["admin.", "unknown", "unemployed", "management", "housemaid",
#                    "entrepreneur", "student", "blue-collar", "self-employed", 
#                    "retired", "technician", "services"]
# attrDict['marital'] = ["married", "divorced", "single"]
# attrDict['education'] = ["unknown", "secondary", "primary", "tertiary"]
# attrDict['default'] = ["yes", "no"]
# attrDict['balance'] = []
# attrDict['housing'] = ["yes", "no"]
# attrDict['loan'] = ["yes", "no"]
# attrDict['contact'] = ["unknown", "telephone", "cellular"]
# attrDict['day'] = []
# attrDict['month'] = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", 
#                      "oct", "nov", "dec"]
# attrDict['duration'] = []
# attrDict['campaign'] = []
# attrDict['pdays'] = []
# attrDict['previous'] = []
# attrDict['poutcome'] = ["unknown", "other", "failure", "success"]
# attrDict['y'] = ["yes", "no"]

# examples_train = DecisionTree.parseCSV(train_data, cols)
# examples_test = DecisionTree.parseCSV(test_data, cols)

# medianList = PreProcess.numerical2binary_MedianThreshold(examples_train, attrDict)
# # use the median of the training data to replace the numerical values of both datasets
# temp, examples_train = PreProcess.numerical2binary_MedianThreshold_Replace(examples_train, attrDict, medianList)
# attrDict, examples_test = PreProcess.numerical2binary_MedianThreshold_Replace(examples_test, attrDict, medianList)

# stumpErrs_train = []
# stumpErrs_test = []

# errorFile = open("bank_errors.csv", 'w')

# print("T\tTraining Data\tTest Data")

# for depth in range(1, maxDepth+1):
#     examples_train, AdaBoostAttrDict = AdaBoost.stringBinaryLabel2numerical(examples_train, attrDict, 'y', 'no', 'yes')
#     examples_test, AdaBoostAttrDict = AdaBoost.stringBinaryLabel2numerical(examples_test, attrDict, 'y', 'no', 'yes')

#     a_list, hyp_list = AdaBoost.AdaBoost(examples_train, AdaBoostAttrDict, 'y', DecisionTree.GainMethods.ENTROPY, depth)

#     predictdata_train = AdaBoost.predict(examples_train, 'prediction', a_list, hyp_list)
#     predictdata_test = AdaBoost.predict(examples_test, 'prediction', a_list, hyp_list)
    
#     if depth == maxDepth:
#         stumpErrs_train = AdaBoost.stumpErrors(examples_train, 'y', 'prediction', hyp_list)
#         stumpErrs_test = AdaBoost.stumpErrors(examples_test, 'y', 'prediction', hyp_list)

#     predictdata_train, oldAttrDict = AdaBoost.numericalLabel2string(predictdata_train, AdaBoostAttrDict, 'y', 'no', 'yes')
#     predictdata_train, oldAttrDict = AdaBoost.numericalLabel2string(predictdata_train, AdaBoostAttrDict, 'prediction', 'no', 'yes')
#     predictdata_test, oldAttrDict = AdaBoost.numericalLabel2string(predictdata_test, AdaBoostAttrDict, 'y', 'no', 'yes')
#     predictdata_test, oldAttrDict = AdaBoost.numericalLabel2string(predictdata_test, AdaBoostAttrDict, 'prediction', 'no', 'yes')

#     total_train = 0
#     wrong_train = 0
#     for example in predictdata_train:
#         if example['y'] != example["prediction"]:
#             wrong_train += 1
#         total_train += 1
#     total_test = 0
#     wrong_test = 0
#     for example in predictdata_test:
#         if example['y'] != example["prediction"]:
#             wrong_test += 1
#         total_test += 1

#     print(f"{depth}\t{wrong_train/total_train:.7f}\t{wrong_test/total_test:.7f}")
#     errorFile.write(f"{depth},{wrong_train/total_train:.7f},{wrong_test/total_test:.7f}\n")

# errorFile.close()

# errorFile = open("bank_stumperrors.csv", 'w')
# print()
# print()
# print("Stump errors at each iteration")
# print("T\tTraining Data\tTest Data")
# for i in range(len(stumpErrs_train)):
#     print(f"{i+1}\t{stumpErrs_train[i]:.7f}\t{stumpErrs_test[i]:.7f}")
#     errorFile.write(f"{i+1},{stumpErrs_train[i]:.7f},{stumpErrs_test[i]:.7f}\n")

# print("Training and Test dataset errors per iteration written to bank_errors.csv")
# print("Decision Stump error for training and test datasets per iteration written to bank_stumperrors.csv")




print()
print()
print()
print("********** Part 2b **********")
print("Bagged Decision Tree experiment")

# the training and test datasets
train_data = "bank/train.csv"
test_data = "bank/test.csv"

maxDepth = 250

# column names
cols = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing',
        'loan', 'contact', 'day', 'month', 'duration', 'campaign', 'pdays',
        'previous', 'poutcome', 'y']

# attribute values
attrDict = {}
attrDict["age"] = []
attrDict['job'] = ["admin.", "unknown", "unemployed", "management", "housemaid",
                   "entrepreneur", "student", "blue-collar", "self-employed", 
                   "retired", "technician", "services"]
attrDict['marital'] = ["married", "divorced", "single"]
attrDict['education'] = ["unknown", "secondary", "primary", "tertiary"]
attrDict['default'] = ["yes", "no"]
attrDict['balance'] = []
attrDict['housing'] = ["yes", "no"]
attrDict['loan'] = ["yes", "no"]
attrDict['contact'] = ["unknown", "telephone", "cellular"]
attrDict['day'] = []
attrDict['month'] = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", 
                     "oct", "nov", "dec"]
attrDict['duration'] = []
attrDict['campaign'] = []
attrDict['pdays'] = []
attrDict['previous'] = []
attrDict['poutcome'] = ["unknown", "other", "failure", "success"]
attrDict['y'] = ["yes", "no"]

examples_train = DecisionTree.parseCSV(train_data, cols)
examples_test = DecisionTree.parseCSV(test_data, cols)

medianList = PreProcess.numerical2binary_MedianThreshold(examples_train, attrDict)
# use the median of the training data to replace the numerical values of both datasets
temp, examples_train = PreProcess.numerical2binary_MedianThreshold_Replace(examples_train, attrDict, medianList)
attrDict, examples_test = PreProcess.numerical2binary_MedianThreshold_Replace(examples_test, attrDict, medianList)

for depth in range(1, maxDepth+1):
    tree_list = BaggedTrees.BaggedDecisionTrees(examples_train, attrDict, 'y', DecisionTree.GainMethods.ENTROPY, depth)

    predictdata_train = BaggedTrees.predict(examples_train, 'prediction', tree_list)
    predictdata_test = BaggedTrees.predict(examples_test, 'prediction', tree_list)
    
    total_train = 0
    wrong_train = 0
    for example in predictdata_train:
        if example['y'] != example["prediction"]:
            wrong_train += 1
        total_train += 1
    total_test = 0
    wrong_test = 0
    for example in predictdata_test:
        if example['y'] != example["prediction"]:
            wrong_test += 1
        total_test += 1

    print(f"{depth}\t{wrong_train/total_train:.7f}\t{wrong_test/total_test:.7f}")




# print()
# print()
# print()
# print("********** Part 2c **********")
# print("Bagged Decision Tree bias and variance experiments")

# TODO: Load data

# for i in range(100):
#     # sample 1000 examples uniformly with replacement from training data

#     # run learn bagged decision trees (500 trees) based on samples

# for each test example, compute prediction of the first tree of each bagged decision tree
# take the average, subtract the ground truth label, and take the square to compute the bias term
# compute the sample variance

# take average over all examples to estimate bias and variance for the single decision tree

# add the two terms to obtain estimate of general square error

# repeat, but used the bagged predictors instead of just the first tree
