##########
# Author: Evan Hrouda
# Purpose: Run various Machine Learning algorithms for the Kaggle competition to determine
#          an individual's income level
##########
import copy
import sys
sys.path.append("../DecisionTree")
sys.path.append("../PreProcess")
sys.path.append("../EnsembleLearning")

import DecisionTree
import PreProcess
import AdaBoost
import BaggedTrees

train_data = "data/train_final.csv"
test_data = "data/test_final.csv"

# column names
cols = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 
        'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 
        'hours-per-week', 'native-country', 'income>50K']
cols_test = ['ID', 'age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 
        'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 
        'hours-per-week', 'native-country']

# attribute values
attrDict = {}
attrDict['age'] = []
attrDict['workclass'] = ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov',
                         'Local-gov', 'State-gov', 'Without-pay', 'Never-worked', '?']
attrDict['fnlwgt'] = []
attrDict['education'] = ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school',
                         'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', 
                         '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool']
attrDict['education-num'] = []
attrDict['marital-status'] = ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated',
                              'Widowed', 'Married-spouse-absent', 'Married-AF-spouse']
attrDict['occupation'] = ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 
                          'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners',
                          'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing',
                          'Transport-moving', 'Priv-house-serv', 'Protective-serv',
                          'Armed-Forces', '?']
attrDict['relationship'] = ['Wife', 'Own-child', 'Husband', 'Not-in-family',
                            'Other-relative', 'Unmarried']
attrDict['race'] = ['White', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other', 'Black']
attrDict['sex'] = ['Female', 'Male']
attrDict['capital-gain'] = []
attrDict['capital-loss'] = []
attrDict['hours-per-week'] = []
attrDict['native-country'] = ['United-States', 'Cambodia', 'England', 'Puerto-Rico', 'Canada',
                              'Germany', 'Outlying-US(Guam-USVI-etc)', 'India', 'Japan', 'Greece',
                              'South', 'China', 'Cuba', 'Iran', 'Honduras', 'Philippines', 'Italy',
                              'Poland', 'Jamaica', 'Vietnam', 'Mexico', 'Portugal', 'Ireland',
                              'France', 'Dominican-Republic', 'Laos', 'Ecuador', 'Taiwan', 'Haiti',
                              'Columbia', 'Hungary', 'Guatemala', 'Nicaragua', 'Scotland',
                              'Thailand', 'Yugoslavia', 'El-Salvador', 'Trinadad&Tobago', 'Peru',
                              'Hong', 'Holand-Netherlands','?']
attrDict['income>50K'] = ['0', '1']
labelCol = 'income>50K'

examples_train = DecisionTree.parseCSV(train_data, cols)[1:]
examples_test = DecisionTree.parseCSV(test_data, cols_test)[1:]

# quartilesList = PreProcess.replaceContinuous_Quartiles(examples_train, attrDict)
# # use the median of the training data to replace the numerical values of both datasets
# temp, examples_train = PreProcess.replaceContinuous_Quartiles_Replace(examples_train, attrDict, quartilesList)
# attrDict, examples_test = PreProcess.replaceContinuous_Quartiles_Replace(examples_test, attrDict, quartilesList)
medianList = PreProcess.numerical2binary_MedianThreshold(examples_train, attrDict)
# use the median of the training data to replace the numerical values of both datasets
temp, examples_train = PreProcess.numerical2binary_MedianThreshold_Replace(examples_train, attrDict, medianList)
attrDict, examples_test = PreProcess.numerical2binary_MedianThreshold_Replace(examples_test, attrDict, medianList)

# for depth in range(1,20):
#     root = DecisionTree.Tree(None)
#     root.depth = 0
#     DecisionTree.ID3(examples_train, attrDict, labelCol, root, depth, DecisionTree.GainMethods.ENTROPY, None)

#     # use the learned tree to predict the label of the training and test datasets
#     predictdata_train = DecisionTree.predict(examples_train, "prediction", root)
#     predictdata_test = DecisionTree.predict(examples_test, "prediction", root)

#     # calculate the error of the training and test dataset
#     total_train = 0
#     wrong_train = 0
#     for example in predictdata_train:
#         if example[labelCol] != example["prediction"]:
#             wrong_train += 1
#         total_train += 1

#     print(f"{depth}\t{wrong_train/total_train:.7f}")

# root = DecisionTree.Tree(None)
# root.depth = 0
# DecisionTree.ID3(examples_train, attrDict, labelCol, root, 6, DecisionTree.GainMethods.ENTROPY, None)

# # use the learned tree to predict the label of the training and test datasets
# predictdata_train = DecisionTree.predict(examples_train, "prediction", root)
# predictdata_test = DecisionTree.predict(examples_test, "prediction", root)

# examples_train, AdaBoostAttrDict = AdaBoost.stringBinaryLabel2numerical(examples_train, attrDict, labelCol, '0', '1')
# examples_test, AdaBoostAttrDict = AdaBoost.stringBinaryLabel2numerical(examples_test, attrDict, labelCol, '0', '1')

# sz = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
sz = [0.1]
for s in sz:
    tree_list = BaggedTrees.BaggedDecisionTrees(examples_train, attrDict, labelCol, DecisionTree.GainMethods.ENTROPY, 500, s)

    predictdata_train = BaggedTrees.predict(examples_train, 'prediction', tree_list)
    predictdata_test = BaggedTrees.predict(examples_test, 'prediction', tree_list)

    total_train = 0
    wrong_train = 0
    for example in predictdata_train:
        if example[labelCol] != example["prediction"]:
            wrong_train += 1
        total_train += 1


    print(f"{s}\t{wrong_train/total_train:.7f}")

with open("predictions.csv",'w') as out:
    out.write("ID,Prediction\n")
    for example in predictdata_test:
        out.write(f"{example['ID']},{example['prediction']}\n")
