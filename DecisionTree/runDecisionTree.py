##########
# Author: Evan Hrouda
# Purpose: Run the ID3 decision tree learning algorithm with the data specified
#          in the homework 1 handout
##########
import DecisionTree
import PreProcess

# 2.2 Run the learning algorithm with car/train.csv dataset then predict the both
# the train.csv and test.csv datasets using all three gain methods and varying the
# tree depth from 1 to 6
print("********** Part 2.2 **********")

# the training and test datasets
train_data = "car/train.csv"
test_data = "car/test.csv"

# column names
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

# very the tree depth up to 6
maxTreeDepth = 6

# parse the csv dataset files
examples_train = DecisionTree.parseCSV(train_data, cols)
examples_test = DecisionTree.parseCSV(test_data, cols)

infoGainErrorData_train = 0
majorityErrorData_train = 0
giniIndexErrorData_train = 0

infoGainErrorData_test = 0
majorityErrorData_test = 0
giniIndexErrorData_test = 0

# using the training dataset to learn a decision tree for each method and tree depths
for gain in DecisionTree.GainMethods:
    for depth in range(1,maxTreeDepth+1):
        root = DecisionTree.Tree(None)
        root.depth = 0
        DecisionTree.ID3(examples_train, cols, attrDict, 'label', root, depth, gain)

        # use the learned tree to predict the label of the training and test datasets
        predictdata_train = DecisionTree.predict(examples_train, attrDict, "prediction", root)
        predictdata_test = DecisionTree.predict(examples_test, attrDict, "prediction", root)

        # calculate the error of the training and test dataset
        total_train = 0
        wrong_train = 0
        for example in predictdata_train:
            if example["label"] != example["prediction"]:
                wrong_train += 1
            total_train += 1
        total_test = 0
        wrong_test = 0
        for example in predictdata_test:
            if example["label"] != example["prediction"]:
                wrong_test += 1
            total_test += 1

        # add the tree depth's error to the full one to calculate the average
        if gain == DecisionTree.GainMethods.ENTROPY:
            infoGainErrorData_train += (wrong_train/total_train)
            infoGainErrorData_test += (wrong_test/total_test)
        elif gain == DecisionTree.GainMethods.MAJORITY:
            majorityErrorData_train += (wrong_train/total_train)
            majorityErrorData_test += (wrong_test/total_test)
        elif gain == DecisionTree.GainMethods.GINI:
            giniIndexErrorData_train += (wrong_train/total_train)
            giniIndexErrorData_test += (wrong_test/total_test)

# average the errors
infoGainErrorData_train /= 6
majorityErrorData_train /= 6
giniIndexErrorData_train /= 6

infoGainErrorData_test /= 6
majorityErrorData_test /= 6
giniIndexErrorData_test /= 6

gains = ["Information Gain", "Majority Error", "Gini Index"]
print("Table of average prediction errors over tree depths 1 to 6 on each dataset")
print("Gain Method\t\tTraining Data\tTest Data")
print(f"Information Gain\t{infoGainErrorData_train:.7f}\t{infoGainErrorData_test:.7f}")
print(f"Majority Error\t\t{majorityErrorData_train:.7f}\t{majorityErrorData_test:.7f}")
print(f"Gini Index\t\t{giniIndexErrorData_train:.7f}\t{giniIndexErrorData_test:.7f}")



# 2.3: Modify ID3 implementation to support numerical attributes. Use a simple approach
# to convert numerical feature to a binary one: choose median of the attribute values as
# the threshold and examine if the feature is bigger or less than the threshold. Use the
# bank dataset (bank/train.csv, bank/test.csv)
print()
print()
print("********** Part 2.3 **********")

# part a: treat "unknown" as a particular attribute value
print("**********  Part a  **********")

# the training and test datasets
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
attrDict['contact'] = [ "unknown", "telephone", "cellular"]
attrDict['day'] = []
attrDict['month'] = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", 
                     "oct", "nov", "dec"]
attrDict['duration'] = []
attrDict['campaign'] = []
attrDict['pdays'] = []
attrDict['previous'] = []
attrDict['poutcome'] = ["unknown", "other", "failure", "success"]
attrDict['y'] = ["yes", "no"]

# very the tree depth up to 16
maxTreeDepth = 16

# parse the csv dataset files
examples_train = DecisionTree.parseCSV(train_data, cols)
examples_test = DecisionTree.parseCSV(test_data, cols)

# find the median of each numeric attribute and convert it to a binary one
attrDict, examples_train = PreProcess.numerical2binary_MedianThreshold(examples_train, cols, attrDict)

gain = DecisionTree.GainMethods.ENTROPY
depth = maxTreeDepth

root = DecisionTree.Tree(None)
root.depth = 0
DecisionTree.ID3(examples_train, cols, attrDict, 'y', root, depth, gain)

predictdata_train = DecisionTree.predict(examples_train, attrDict, "prediction", root)
# predictdata_test = DecisionTree.predict(examples_test, attrDict, "prediction", root)

total_train = 0
wrong_train = 0
for example in predictdata_train:
    if example["y"] != example["prediction"]:
        wrong_train += 1
    total_train += 1

print(f"{wrong_train} / {total_train} = {wrong_train/total_train}")
