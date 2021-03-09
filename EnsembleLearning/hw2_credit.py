##########
# Author: Evan Hrouda
# Purpose: Perform AdaBoost, Bagged Trees, and Random Forests on the credit dataset
##########
import math
import sys
sys.path.append("../DecisionTree")
sys.path.append("../PreProcess")

import DecisionTree
import PreProcess
import BaggedTrees
import RandomForest
import AdaBoost

print()
print()
print()
print("********** Part 3  **********")
print("Bagged Decision Tree experiment")

# the training and test datasets
# data = "credit/credit.csv"
examples_train = "credit/train.csv"
examples_test = "credit/test.csv"

maxDepth = 500

# column names
cols = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6',
        'BILL_AMT1', 'BILL_AMT2', 'BILL_AMT3', 'BILL_AMT4', 'BILL_AMT5', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2',
        'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6', 'default payment next month']
labelCol = 'default payment next month'

# attribute values
attrDict = {}
attrDict['LIMIT_BAL'] = []
attrDict['SEX'] = ['1', '2']
attrDict['EDUCATION'] = ['0', '1', '2', '3', '4', '5', '6']
attrDict['MARRIAGE'] = ['0', '1', '2', '3']
attrDict['AGE'] = []
attrDict['PAY_0'] = ['-2','-1', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
attrDict['PAY_2'] = ['-2','-1', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
attrDict['PAY_3'] = ['-2','-1', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
attrDict['PAY_4'] = ['-2','-1', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
attrDict['PAY_5'] = ['-2','-1', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
attrDict['PAY_6'] = ['-2','-1', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
attrDict['BILL_AMT1'] = []
attrDict['BILL_AMT2'] = []
attrDict['BILL_AMT3'] = []
attrDict['BILL_AMT4'] = []
attrDict['BILL_AMT5'] = []
attrDict['BILL_AMT6'] = []
attrDict['PAY_AMT1'] = []
attrDict['PAY_AMT2'] = []
attrDict['PAY_AMT3'] = []
attrDict['PAY_AMT4'] = []
attrDict['PAY_AMT5'] = []
attrDict['PAY_AMT6'] = []
attrDict['default payment next month']  = ['0', '1']

# data = DecisionTree.parseCSV(data, cols)
# Randomly choose 24000 examples to be training data, remaing 6000 are test
# examples_train = []
# examples_test = []

# while len(examples_train) != 24000:
#     index = random.randrange(len(data))
#     examples_train.append(data[index])
#     del data[index]
# examples_test = data

# with open("credit/train.csv", 'w') as f:
#     for ex in examples_train:
#         f.write(f"{ex['LIMIT_BAL']},{ex['SEX']},{ex['EDUCATION']},{ex['MARRIAGE']},"
#                 f"{ex['AGE']},{ex['PAY_0']},{ex['PAY_2']},{ex['PAY_3']},{ex['PAY_4']},"
#                 f"{ex['PAY_5']},{ex['PAY_6']},{ex['BILL_AMT1']},{ex['BILL_AMT2']},"
#                 f"{ex['BILL_AMT3']},{ex['BILL_AMT4']},{ex['BILL_AMT5']},{ex['BILL_AMT6']},"
#                 f"{ex['PAY_AMT1']},{ex['PAY_AMT2']},{ex['PAY_AMT3']},{ex['PAY_AMT4']},"
#                 f"{ex['PAY_AMT5']},{ex['PAY_AMT6']},{ex['default payment next month']}\n")
# with open("credit/test.csv", 'w') as f:
#     for ex in examples_test:
#         f.write(f"{ex['LIMIT_BAL']},{ex['SEX']},{ex['EDUCATION']},{ex['MARRIAGE']},"
#                 f"{ex['AGE']},{ex['PAY_0']},{ex['PAY_2']},{ex['PAY_3']},{ex['PAY_4']},"
#                 f"{ex['PAY_5']},{ex['PAY_6']},{ex['BILL_AMT1']},{ex['BILL_AMT2']},"
#                 f"{ex['BILL_AMT3']},{ex['BILL_AMT4']},{ex['BILL_AMT5']},{ex['BILL_AMT6']},"
#                 f"{ex['PAY_AMT1']},{ex['PAY_AMT2']},{ex['PAY_AMT3']},{ex['PAY_AMT4']},"
#                 f"{ex['PAY_AMT5']},{ex['PAY_AMT6']},{ex['default payment next month']}\n")


examples_train = DecisionTree.parseCSV(examples_train, cols)
examples_test = DecisionTree.parseCSV(examples_test, cols)

medianList = PreProcess.numerical2binary_MedianThreshold(examples_train, attrDict)
# use the median of the training data to replace the numerical values of both datasets
temp, examples_train = PreProcess.numerical2binary_MedianThreshold_Replace(examples_train, attrDict, medianList)
attrDict, examples_test = PreProcess.numerical2binary_MedianThreshold_Replace(examples_test, attrDict, medianList)

print("T\tTraining Data\tTest Data")
for depth in range(1, maxDepth+1):
    tree_list = BaggedTrees.BaggedDecisionTrees(examples_train, attrDict, labelCol, DecisionTree.GainMethods.ENTROPY, depth, 0.4)

    predictdata_train = BaggedTrees.predict(examples_train, 'prediction', tree_list)
    predictdata_test = BaggedTrees.predict(examples_test, 'prediction', tree_list)
    
    total_train = 0
    wrong_train = 0
    for example in predictdata_train:
        if example[labelCol] != example["prediction"]:
            wrong_train += 1
        total_train += 1
    total_test = 0
    wrong_test = 0
    for example in predictdata_test:
        if example[labelCol] != example["prediction"]:
            wrong_test += 1
        total_test += 1

    print(f"{depth}\t{wrong_train/total_train:.7f}\t{wrong_test/total_test:.7f}")
    with open("credit_errors_bagged.csv", 'a') as errorFile:
        errorFile.write(f"{depth},{wrong_train/total_train:.7f},{wrong_test/total_test:.7f}\n")

print("Training and Test dataset errors per number of trees written to credit_errors_bagged.csv")




print()
print()
print()
print("********** Part 3  **********")
print("Random Forests experiment")

# vary number of random trees from 1 to 500 and vary feature subset from {2, 4, 6}
subsetSizes = [2, 4, 6]
print("Size\tT\tTraining Data\tTest Data")
for sz in subsetSizes:
    for depth in range(1, maxDepth+1):
        tree_list = RandomForest.RandomForests(examples_train, attrDict, labelCol, DecisionTree.GainMethods.ENTROPY, depth, sz, 0.4)

        predictdata_train = RandomForest.predict(examples_train, 'prediction', tree_list)
        predictdata_test = RandomForest.predict(examples_test, 'prediction', tree_list)
    
        total_train = 0
        wrong_train = 0
        for example in predictdata_train:
            if example[labelCol] != example["prediction"]:
                wrong_train += 1
            total_train += 1
        total_test = 0
        wrong_test = 0
        for example in predictdata_test:
            if example[labelCol] != example["prediction"]:
                wrong_test += 1
            total_test += 1

        print(f"{sz}\t{depth}\t{wrong_train/total_train:.7f}\t{wrong_test/total_test:.7f}")
        with open(f"credit_errors_randforests_featsz{sz}.csv", 'a') as errorFile:
            errorFile.write(f"{depth},{wrong_train/total_train:.7f},{wrong_test/total_test:.7f}\n")

print("Training and Test dataset errors per number of trees written to credit_errors_randforests_featsz[sz].csv")
print("where [sz] is the size of the number of features randomly selected in tree creation.")



print()
print()
print()
print("********** Part 3  **********")
print("AdaBoost experiment")

print("T\tTraining Data\tTest Data")

for depth in range(1, maxDepth+1):
    examples_train, AdaBoostAttrDict = AdaBoost.stringBinaryLabel2numerical(examples_train, attrDict, labelCol, '0', '1')
    examples_test, AdaBoostAttrDict = AdaBoost.stringBinaryLabel2numerical(examples_test, attrDict, labelCol, '0', '1')

    a_list, hyp_list = AdaBoost.AdaBoost(examples_train, AdaBoostAttrDict, labelCol, DecisionTree.GainMethods.ENTROPY, depth)

    predictdata_train = AdaBoost.predict(examples_train, 'prediction', a_list, hyp_list)
    predictdata_test = AdaBoost.predict(examples_test, 'prediction', a_list, hyp_list)
    
    if depth == maxDepth:
        stumpErrs_train = AdaBoost.stumpErrors(examples_train, labelCol, 'prediction', hyp_list)
        stumpErrs_test = AdaBoost.stumpErrors(examples_test, labelCol, 'prediction', hyp_list)

    predictdata_train, oldAttrDict = AdaBoost.numericalLabel2string(predictdata_train, AdaBoostAttrDict, labelCol, '0', '1')
    predictdata_train, oldAttrDict = AdaBoost.numericalLabel2string(predictdata_train, AdaBoostAttrDict, 'prediction', '0', '1')
    predictdata_test, oldAttrDict = AdaBoost.numericalLabel2string(predictdata_test, AdaBoostAttrDict, labelCol, '0', '1')
    predictdata_test, oldAttrDict = AdaBoost.numericalLabel2string(predictdata_test, AdaBoostAttrDict, 'prediction', '0', '1')

    total_train = 0
    wrong_train = 0
    for example in predictdata_train:
        if example[labelCol] != example["prediction"]:
            wrong_train += 1
        total_train += 1
    total_test = 0
    wrong_test = 0
    for example in predictdata_test:
        if example[labelCol] != example["prediction"]:
            wrong_test += 1
        total_test += 1

    print(f"{depth}\t{wrong_train/total_train:.7f}\t{wrong_test/total_test:.7f}")
    with open("credit_errors_adaboost.csv", 'a') as errorFile:
        errorFile.write(f"{depth},{wrong_train/total_train:.7f},{wrong_test/total_test:.7f}\n")\

print("Training and Test dataset errors per number of trees written to credit_errors_adaboost.csv")




print()
print()
print()
print("********** Part 3  **********")
print("Single Decision Tree experiment")

print("T\tTraining Data\tTest Data")
root = DecisionTree.Tree(None)
root.depth = 0
DecisionTree.ID3(examples_train, attrDict, labelCol, root, math.inf, DecisionTree.GainMethods.ENTROPY, None)

# use the learned tree to predict the label of the training and test datasets
predictdata_train = DecisionTree.predict(examples_train, "prediction", root)
predictdata_test = DecisionTree.predict(examples_test, "prediction", root)

# calculate the error of the training and test dataset
total_train = 0
wrong_train = 0
for example in predictdata_train:
    if example[labelCol] != example["prediction"]:
        wrong_train += 1
    total_train += 1
total_test = 0
wrong_test = 0
for example in predictdata_test:
    if example[labelCol] != example["prediction"]:
        wrong_test += 1
    total_test += 1

print(f"{wrong_train/total_train:.7f}\t{wrong_test/total_test:.7f}")
