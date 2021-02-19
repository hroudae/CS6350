##########
# Author: Evan Hrouda
# Purpose: Run the ID3 decision tree learning algorithm with the data specified
#          in the homework 1 handout
##########
import DecisionTree
import PreProcess

#####
# Author: Evan Hrouda
# Purpose: run through the ID3 algorithm for each gain method and up to the specified
#          max tree depth. Print out the average error for the training and test data
#          for each method.
#####
def createTreeAndPredict(train_data, test_data, cols, attrDict, labelCol, maxTreeDepth, numerical2binary, replaceUnknown, unknown):
    # parse the csv dataset files
    examples_train = DecisionTree.parseCSV(train_data, cols)
    examples_test = DecisionTree.parseCSV(test_data, cols)

    if numerical2binary:
        temp, examples_train = PreProcess.numerical2binary_MedianThreshold(examples_train, attrDict)
        attrDict, examples_test = PreProcess.numerical2binary_MedianThreshold(examples_test, attrDict)

    if replaceUnknown:
        examples_train = PreProcess.replaceUnknown_MajorityAttribute(examples_train, attrDict, unknown)
        examples_test = PreProcess.replaceUnknown_MajorityAttribute(examples_test, attrDict, unknown)

    infoGainErrorData_train = 0
    majorityErrorData_train = 0
    giniIndexErrorData_train = 0

    infoGainErrorData_test = 0
    majorityErrorData_test = 0
    giniIndexErrorData_test = 0

    # using the training dataset to learn a decision tree for each method and tree depths
    for gain in DecisionTree.GainMethods:
        if gain == DecisionTree.GainMethods.ENTROPY:
            print("Table of prediction errors at each depth using gain method Information Gain")
        elif gain == DecisionTree.GainMethods.GINI:
            print("Table of prediction errors at each depth using gain method Gini Index")
        elif gain == DecisionTree.GainMethods.MAJORITY:
            print("Table of prediction errors at each depth using gain method Majority Error")
        print("Depth\tTraining Data\tTest Data")

        for depth in range(1,maxTreeDepth+1):
            root = DecisionTree.Tree(None)
            root.depth = 0
            DecisionTree.ID3(examples_train, attrDict, labelCol, root, depth, gain)

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

            print(f"{depth}\t{wrong_train/total_train:.7f}\t{wrong_test/total_test:.7f}")

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
        print() # new line in between gain methods

    # average the errors
    infoGainErrorData_train /= maxTreeDepth
    majorityErrorData_train /= maxTreeDepth
    giniIndexErrorData_train /= maxTreeDepth

    infoGainErrorData_test /= maxTreeDepth
    majorityErrorData_test /= maxTreeDepth
    giniIndexErrorData_test /= maxTreeDepth

    gains = ["Information Gain", "Majority Error", "Gini Index"]
    print(f"Table of average prediction errors over tree depths 1 to {maxTreeDepth} on each dataset")
    print("Gain Method\t\tTraining Data\tTest Data")
    print(f"Information Gain\t{infoGainErrorData_train:.7f}\t{infoGainErrorData_test:.7f}")
    print(f"Majority Error\t\t{majorityErrorData_train:.7f}\t{majorityErrorData_test:.7f}")
    print(f"Gini Index\t\t{giniIndexErrorData_train:.7f}\t{giniIndexErrorData_test:.7f}")

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

# run the tests as specified in homework 1 part 2.2b
createTreeAndPredict(train_data, test_data, cols, attrDict, 'label', 6, False, False, None)



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

# run the tests as specified in homework 1 part 2.3a
createTreeAndPredict(train_data, test_data, cols, attrDict, 'y', 16, True, False, None)


# part b: treat "unknown" as a missing attribute value - replace it with the majority value
print()
print()
print("**********  Part b  **********")
createTreeAndPredict(train_data, test_data, cols, attrDict, 'y', 16, True, True, "unknown")
