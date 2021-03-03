# CS6350

This is a machine learning library developed by Evan Hrouda for CS5350/6350 at the University of Utah.

## Decision Tree

The implementation of the ID3 algorithm is found in the DecisionTree folder. DecisionTree.py contains the implementation.

To obtain the properly formatted example list, the function DecisionTree.parseCSV(csvFilePath, cols) is called.

Inputs:

- csvFilePath: a string that is the path to a .csv file containing the example data with no headers and delimited by a comma
- cols: a list of the attribute column names in the order the data in the .csv file appears in

Ouput:

- data: a list of dictionaries where each example is a dictionary whose keys are the attribute column names and the values are the values for that example

To learn a decison tree, the function DecisionTree.ID3(data, attrDict, labelCol, node, maxDepth, gainMethod, D) is called.

Inputs:

- data: a list of dictionaries where each dictionary corresponds to one example. The keys of the dictionaries are the attribute column names and the values are the feature values of the example. This list is created by calling DecisionTree.parseCSV(csvFilePath, cols)
- attrDict: a dictionary where the keys are the attribute names and the values are all the possible values for that attribute
- labelCol: a string which is the name of the label column in the training data
- node: a root DecisionTree.Tree(None) node with depth set to 0 on which the decision tree is built
- maxDepth: The maxium depth of the decision tree
- gainMethod: the gain method to use when splitting the data. The DecisionTree.GainMethods are ENTROPY for information gain, GINI for gini index, and MAJORITY for majority error.
- D: a list of weights corresponding to the index of the example in data. If D is None, weights are assumed to be 1

Output:

- The root node passed in as node is the root of the tree

To predict labels, use DecisionTree.predict(data, predictCol, root).

Inputs:

- data: a list of dictionaries where each dictionary corresponds to one example. The keys of the dictionaries are the attribute column names and the values are the feature values of the example. This list is created by calling DecisionTree.parseCSV(csvFilePath, cols)
- predictCol: the name of the column where the label prediction should be added to in the example dictionary in the data list
- root: the root node of the tree obtained by running the ID3 algorithm

Output:
- data: the same example list passed in with each example having an additional key,value pair where the key is predictCol and the value is the decision tree's label prediction


There are two pre-processing techniques implemented in PreProcess.py

PreProcess.numerical2binary_MedianThreshold is used to replace numerical attributes with their relationship to the median of that attribute.

PreProcess.numerical2binary_MedianThreshold(data, attrDict) is used to find the median of the numerical attributes.

Inputs:

- data: the example list genereated from DecisionTree.parseCSV(csvFilePath, cols)
- attrDict: a dictionary where the keys are the attribute names and the values are a list of all the possible values for that attribute. Numerical attributes should have an empty list as their value.

Output:

- medianList: a dictionary where the keys are the numerical attributes and the values are the median for that attribute

numerical2binary_MedianThreshold_Replace(data, attrDict, medianList) is used to replace the numerical values with their relationship to the median. Those larger than the median are given the label "larger" and those equal to or smaller than the median get the label "smaller"

Inputs:

- data: the example list genereated from DecisionTree.parseCSV(csvFilePath, cols)
- attrDict: a dictionary where the keys are the attribute names and the values are a list of all the possible values for that attribute. 
- medianList: the dictionary of attribute medians obtained from reProcess.numerical2binary_MedianThreshold().


The other preprocessing technique implemented is to replace unknown attributes with the majority attribute for that attribute. There are two functions that must be called:

replaceUnknown_MajorityAttribute(data, attrDict, unknown) is used to find the majority attribute value for all attributes (without counting the unknown values)

Inputs:

- data: the example list genereated from DecisionTree.parseCSV(csvFilePath, cols)
- attrDict: a dictionary where the keys are the attribute names and the values are a list of all the possible values for that attribute. 
- unknown: a string that is the unknown value in the example data that must replaced

Output:

- majorityAttrs: a dictionary where the keys are the attribute names and the values are the majority value for that attribute

replaceUnknown_MajorityAttribute_Replace(data, majorityAttrs, unknown) is used to replace the unknown values with the majority one

Inputs:

- data: the example list genereated from DecisionTree.parseCSV(csvFilePath, cols)
- majorityAttrs: a dictionary where the keys are the attribute names and the values are the majority value for that attribute 
- unknown: a string that is the unknown value in the example data that must replaced

Output:

- data: the example list passed in with all the unkown values replaced with the majority attribute value



An example of how to learn a decision tree and then predict labels, along with how the two preprocessing techniques are used, can be found in the file runDecisionTree.py.
