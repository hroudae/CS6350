##########
# Author: Evan Hrouda
# Purpose: Perform the required experiments as described in Homework 2 handout
#          for Random Forests
##########
import random
import statistics
import sys
sys.path.append("../DecisionTree")
sys.path.append("../PreProcess")

import DecisionTree
import PreProcess
import RandomForest

print()
print()
print()
print("********** Part 2d **********")
print("Random Forest experiment")

# the training and test datasets
train_data = "bank/train.csv"
test_data = "bank/test.csv"

maxDepth = 500

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

# vary number of random trees from 1 to 500 and vary feature subset from {2, 4, 6}
subsetSizes = [2, 4, 6]
print("Size\tT\tTraining Data\tTest Data")
for sz in subsetSizes:
    for depth in range(1, maxDepth+1):
        tree_list = RandomForest.RandomForests(examples_train, attrDict, 'y', DecisionTree.GainMethods.ENTROPY, depth, sz, 0.4)

        predictdata_train = RandomForest.predict(examples_train, 'prediction', tree_list)
        predictdata_test = RandomForest.predict(examples_test, 'prediction', tree_list)
    
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

        print(f"{sz}\t{depth}\t{wrong_train/total_train:.7f}\t{wrong_test/total_test:.7f}")
        with open(f"bank_errors_randforests_featsz{sz}.csv", 'a') as errorFile:
            errorFile.write(f"{depth},{wrong_train/total_train:.7f},{wrong_test/total_test:.7f}\n")

print("Training and Test dataset errors per number of trees written to bank_errors_randforests_featsz[sz].csv")
print("where [sz] is the size of the number pf features randomly selected in tree creation.")





print()
print()
print()
print("********** Part 2e **********")
print("Random Forests bias and variance experiments")

train_data = "bank/train.csv"
test_data = "bank/test.csv"

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

trees = []
for i in range(100):
    # sample 1000 examples uniformly with replacement from training data
    samples = random.sample(examples_train, 1000)
    # run learn random forests (500 trees) based on samples
    tree_list = RandomForest.RandomForests(samples, attrDict, 'y', DecisionTree.GainMethods.ENTROPY, 500, 4, 0.4)
    trees.append(tree_list)

# for each test example, compute prediction of the first tree of each bagged decision tree
# take the average, subtract the ground truth label, and take the square to compute the bias term
# compute the sample variance
singleTreeBiases = []
singleTreeVariance = []
for example in examples_test:
    predictionList = []
    for tree_list in trees:
        prediction = DecisionTree.predict_example(example, 'prediction', tree_list[0])
        if prediction == 'yes':
            predictionList.append(1)
        else:
            predictionList.append(-1)
    bias = 0
    if example['y'] == 'yes':
        bias = 1
    else:
        bias = -1
    predictionAvg = statistics.mean(predictionList)
    bias -= predictionAvg
    bias *= bias
    singleTreeBiases.append(bias)

    variance = statistics.variance(predictionList)
    singleTreeVariance.append(variance)

# take average over all examples to estimate bias and variance for the single decision tree
biasEst = statistics.mean(singleTreeBiases)
varEst = statistics.mean(singleTreeVariance)
# add the two terms to obtain estimate of general square error
genSquareErrEst = biasEst + varEst
print(f"Single Decision Tree: General Bias = {biasEst:.7f}, General Variance = {varEst:.7f}, General Squared Error = {genSquareErrEst:.7f}")


# repeat, but used the bagged predictors instead of just the first tree
randomForestBiases = []
randomForestVariance = []
for example in examples_test:
    predictionList = []
    for tree_list in trees:
        prediction = RandomForest.predict_example(example, "prediction", tree_list)
        if prediction == 'yes':
            predictionList.append(1)
        else:
            predictionList.append(-1)
    bias = 0
    if example['y'] == 'yes':
        bias = 1
    else:
        bias = -1
    predictionAvg = statistics.mean(predictionList)
    bias -= predictionAvg
    bias *= bias
    randomForestBiases.append(bias)

    variance = statistics.variance(predictionList)
    randomForestVariance.append(variance)

# take average over all examples to estimate bias and variance for the single decision tree
biasEst = statistics.mean(randomForestBiases)
varEst = statistics.mean(randomForestVariance)
# add the two terms to obtain estimate of general square error
genSquareErrEst = biasEst + varEst
print(f"Random Forest: General Bias = {biasEst:.7f}, General Variance = {varEst:.7f}, General Squared Error = {genSquareErrEst:.7f}")
