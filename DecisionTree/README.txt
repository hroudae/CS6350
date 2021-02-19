To run the program to produce the tables specified in homework 1 part 2, run
the shell script run.sh. This will average the prediction errors over the
varied tree depths for each gain method and the two datasets as well as print
the prediction error at each depth. The program that the shell script runs is
runDecisionTree.py.

Implementation details:
The ID3 algorithm is implemented in DecicionTree.py:
ID3(data, attrDict, labelCol, node, maxDepth, gainMethod)
where data is a list of examples produced using the parseCSV function, attrDict
is a dictionary with attribute names as the key and a list of values as the value,
labelCol is the name of the label attribute column in the training data, node is
the node of the tree to be split on next, maxDepth is the maximum depth the tree
should be, and gainMethod is a GainMethods enum value to specify which gain method
should be used to calculate purity and gain.

The function to predict the label of examples is implemented in DecicionTree.py:
predict(data, predictCol, root)
where data is a list of data to predict the label of, predictCol is the name of
the key where the prediction will be put in the example in the data, and root is
the root of the decision tree produced by the ID3 function.

PreProcess.py has two main preprocssing functions:
numerical2binary_MedianThreshold(data, attrDict)
converts numerical attribute values to binary ones by splitting over the median of
the data. The values bigger than the median will be labeled "larger" and the ones
equal to or smaller than the median will be labelled "smaller".

replaceUnknown_MajorityAttribute(data, attrDict, unknown)
replaces unknown values with the majority value of the attribute.

Other helper functions used in the ID3 implementation:
best(data, attrList, labelCol, gainMethod)
is implemented in DecicionTree.py and finds the attribute with the highest gain using
the specified gain method.

purity(val, attr, data, labelList, labelCol, gainMethod)
is implemented in DecicionTree.py and calculates the purity of the attribute using
the specified gain method.
