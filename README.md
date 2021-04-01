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



## AdaBoost

The implementation of the AdaBoost algorithm is found in the EnsembleLearning folder. AdaBoost.py contains the implementation.

To obtain the properly formatted example list, the function DecisionTree.parseCSV(csvFilePath, cols) is called as described above. Additionally, any binary labels must be converted to numerical values -1 and 1.

stringBinaryLabel2numerical(data, attrDict, labelCol, negVal, posVal) will convert binary labels to numerical ones.

Inputs: 

- data: the example list genereated from DecisionTree.parseCSV(csvFilePath, cols)
- attrDict: a dictionary where the keys are the attribute names and the values are a list of all the possible values for that attribute.
- labelCol: a string which is the name of the label column in the training data
- negVal: value of the binary label that should be converted to -1
- posVal: value of the binary label that should be converted to 1

Outputs:

- newData: the example list with label column values replaced
- newAttrDict: the new attribute dictionary with the binary label values replaced

numericalLabel2string(data, attrDict, labelCol, negVal, posVal) will convert the numerical label back to the desired binary label. 

Inputs: 

- data: the example list genereated from DecisionTree.parseCSV(csvFilePath, cols) with numerical label values {-1, 1}
- attrDict: a dictionary where the keys are the attribute names and the values are a list of all the possible values for that attribute.
- labelCol: a string which is the name of the label column in the training data
- negVal: value of the binary label that should be converted to from -1
- posVal: value of the binary label that should be converted to from 1

Outputs:

- newData: the example list with label column values replaced
- newAttrDict: the new attribute dictionary with the binary label


AdaBoost(data, attrDict, labelCol, gainMethod, T) implements the AdaBoost learning algorithm

Inputs:

- data: the example list genereated from DecisionTree.parseCSV(csvFilePath, cols) with numerical label values {-1, 1}
- attrDict: a dictionary where the keys are the attribute names and the values are a list of all the possible values for that attribute.
- labelCol: a string which is the name of the label column in the training data
- gainMethod: the gain method to use when splitting the data. The DecisionTree.GainMethods are ENTROPY for information gain, GINI for gini index, and MAJORITY for majority error.
- T: the number of decision stumps to learn

Outputs:

- a_list: the vote list for each decision stump
- hyp_list: a list of decision stumps generated by the DecisionTree.ID3 algorithm. The stump's vote is located at the same index in a_list


predict(data, predictCol, a_list, hyp_list) uses the vote list and stump list generated by AdaBoost() to predict the data

Inputs:

- data: the example list genereated from DecisionTree.parseCSV(csvFilePath, cols) to be predicted 
- predictCol: the name of the column where the label prediction should be added to in the example dictionary in the data list 
- a_list: the vote list from AdaBoost()
- hyp_list the decision stump list from AdaBoost()

Outputs:

predictData: the example list with a new column for the numerical prediction of the examples


An example of how to use AdaBoost can be found in the file EnsembleLearning/hw2_adaboost.py



## Bagged Decision Trees

The implementation of the Bagged Decision Trees algorithm is found in the EnsembleLearning folder. BaggedTrees.py contains the implementation.

To obtain the properly formatted example list, the function DecisionTree.parseCSV(csvFilePath, cols) is called as described above.

BaggedDecisionTrees(data, attrDict, labelCol, gainMethod, T, samplesize) implements the Bagged Decision Tree learning algorithm.

Inputs:

- data: the example list genereated from DecisionTree.parseCSV(csvFilePath, cols)
- attrDict: a dictionary where the keys are the attribute names and the values are a list of all the possible values for that attribute.
- labelCol: a string which is the name of the label column in the training data
- gainMethod: the gain method to use when splitting the data. The DecisionTree.GainMethods are ENTROPY for information gain, GINI for gini index, and MAJORITY for majority error.
- T: the number of iterations to run
- samplesize: the percentage in decimal form of the data examples that should drawn uniformly random with replacement feach iteration

Outputs:

- trees: a list of decision trees that can be used to predict labels


predict(data, predictCol, trees) uses purality vote to predict labels for the data

Inputs:

- data: the example list genereated from DecisionTree.parseCSV(csvFilePath, cols) with numerical label values {-1, 1} to be predicted 
- predictCol: the name of the column where the label prediction should be added to in the example dictionary in the data list 
- trees: the list of trees to use generatedd by BaggedDecisionTrees()

Outputs:

- predictData: the example list with a new column containing the label prediction


An example of how to use BaggedDecisionTrees can be found in the file EnsembleLearning/hw2_baggedtrees.py




## Random Forests

The implementation of the Random Forests algorithm is found in the EnsembleLearning folder. RandomForest.py contains the implementation.

To obtain the properly formatted example list, the function DecisionTree.parseCSV(csvFilePath, cols) is called as described above.

RandomForests(data, attrDict, labelCol, gainMethod, T, featureSetSize, samplesize) implements the learning algorithm

Inputs:

- data: the example list genereated from DecisionTree.parseCSV(csvFilePath, cols)
- attrDict: a dictionary where the keys are the attribute names and the values are a list of all the possible values for that attribute.
- labelCol: a string which is the name of the label column in the training data
- gainMethod: the gain method to use when splitting the data. The DecisionTree.GainMethods are ENTROPY for information gain, GINI for gini index, and MAJORITY for majority error.
- T: the number of iterations to run
- featureSetSize: the number of attributes to sample when constructing the decision trees
- samplesize: the percentage in decimal form of the data examples that should drawn uniformly random with replacement feach iteration

Outputs:

- forest: a list of decision trees that can be used to predict labels


predict(data, predictCol, forest) uses purality vote to predict labels for the data

Inputs:

- data: the example list genereated from DecisionTree.parseCSV(csvFilePath, cols) with numerical label values {-1, 1} to be predicted 
- predictCol: the name of the column where the label prediction should be added to in the example dictionary in the data list 
- forest: the list of trees to use generatedd by RandomForests()

Outputs:

- predictData: the example list with a new column containing the label prediction


An example of how to use RandomForests can be found in the file EnsembleLearning/hw2_randomforests.py




## Batch Gradient Descent LMS

The implementation of the Batch Gradient Descent algorithm is found in the LinearRegression folder. BatchGradientDescent.py contains the implementation.

To obtain the properly formatted example list, the function DecisionTree.parseCSV(csvFilePath, cols) is called as described above. Then createInputMatrices(data, labelCol) from utilities.py is called to generate the required matrices.

createInputMatrices(data, labelCol) generates the required augmented x and y matrices.

Inputs:

- data: the example list genereated from DecisionTree.parseCSV(csvFilePath, cols)
- labelCol: a string which is the name of the label column in the training data

Outputs:

- x: the augmented x matrix to be used with either linear regression algorithm
- y: the y vector that contains the label data for each example in x


GradientDescent(x, y, r) implements the learning algorithm

Inputs:

- x: the x matrix generated by createInputMatrices()
- y: the y vector generated by createInputMatrices()
- r: the learning rate to use

Outputs:

- wghts: the learned weight vector
- costs: a list of cost function values at each iteration


An example of how to use GradientDescent can be found in the file LinearRegression/hw2_gradients.py




## Stochastic Gradient Descent LMS

The implementation of the Stochastic Gradient Descent algorithm is found in the LinearRegression folder. StochasticGradientDescent.py contains the implementation.

To obtain the properly formatted example list, the function DecisionTree.parseCSV(csvFilePath, cols) is called as described above. Then createInputMatrices(data, labelCol) from utilities.py is called to generate the required matrices.

StochasticGradientDescent(x, y, r, iterations) implements the learning algorithm

Inputs:

- x: the x matrix generated by createInputMatrices()
- y: the y vector generated by createInputMatrices()
- r: the learning rate to use
- iterations: the number of iterations to run the algorithm for

Outputs:

- wghts: the learned weight vector
- costs: a list of cost function values at each iteration 
- converge: a boolean value that says whether or not the algorithm converged

An example of how to use StochasticGradientDescent can be found in the file LinearRegression/hw2_gradients.py




## Perceptron

To obtain the properly formatted example list for all Perceptron algorithms, the function Perceptron.parseCSV(csvFilePath, zero2neg) in Perceptron.py should be called.

Perceptron.parseCSV(csvFilePath, zero2neg) creates the correct x matrix and y vector from the provided csv file.

Inputs:

- csvFilePath: a string that is the path to a .csv file containing the example data with no headers and delimited by a comma. All values in the csv should be real numbers.
- zero2neg: A boolean value. True indicates that the label column is in the range {0, 1} which must be converted to the range {-1, 1}. A False value will skip the conversion.

Outputs:

- x: a numpy matrix where each row is an example's attribute value as a float
- y: a numpy vector where each element is the corresponding row in x's label value as a float

For csv files containg data with no label column, such as test data, parseCSV_NoLabel(csvFilePath) is used to generate just the proper example matrix x.



## Standard Perceptron

The implementation of the standard Perceptron algorithm is found in the Perceptron folder. Perceptron.py contains the implementation.

To obtain the properly formatted x and y, the function Perceptron.parseCSV(csvFilePath, zero2neg) is called as described above.

StandardPerceptron(x, y, r, T) implements the learning algorithm

Inputs:

- x: the x matrix generated by parseCSV(csvFilePath, zero2neg) 
- y: the y vector generated by parseCSV(csvFilePath, zero2neg) 
- r: the learning rate to use
- T: the number of epochs to run the algorithm for

Outputs:

- wghts: the learned weight vector


predict_StandardPerceptron(x, w) predicts the labels for each row of x using the weight vector w.

Inputs:

- x: the example list genereated from parseCSV_NoLabel(csvFilePath) or parseCSV(csvFilePath, zero2neg)
- w: the learned weight vector to be used to predict the label of the examples

Outputs:

- predictions: a numpy vector where each element is the predicted label in {-1, 1} for the corresponding example

An example of how to use StandardPerceptron can be found in the file Perceptron/hw3_perceptron.py




## Voted Perceptron

The implementation of the voted Perceptron algorithm is found in the Perceptron folder. Perceptron.py contains the implementation.

To obtain the properly formatted x and y, the function Perceptron.parseCSV(csvFilePath, zero2neg) is called as described above.

VotedPerceptron(x, y, r, T) implements the learning algorithm

Inputs:

- x: the x matrix generated by parseCSV(csvFilePath, zero2neg) 
- y: the y vector generated by parseCSV(csvFilePath, zero2neg) 
- r: the learning rate to use
- T: the number of epochs to run the algorithm for

Outputs:

- wght_list: a list of tuples where the first element of the tuple is the learned weight vector and the second is that weight vector's count


predict_VotedPerceptron(x, wght_list) predicts the labels for each row of x using the weight vectors and counts found in wght_list

Inputs:

- x: the example list genereated from parseCSV_NoLabel(csvFilePath) or parseCSV(csvFilePath, zero2neg)
- wght_list: a list of tuples where the first element of the tuple is the learned weight vector and the second is that weight vector's count to be used to predict the label of the examples

Outputs:

- predictions: a numpy vector where each element is the predicted label in {-1, 1} for the corresponding example

An example of how to use VotedPerceptron can be found in the file Perceptron/hw3_perceptron.py




## Averaged Perceptron

The implementation of the averaged Perceptron algorithm is found in the Perceptron folder. Perceptron.py contains the implementation.

To obtain the properly formatted x and y, the function Perceptron.parseCSV(csvFilePath, zero2neg) is called as described above.

AveragedPerceptron(x, y, r, T) implements the learning algorithm

Inputs:

- x: the x matrix generated by parseCSV(csvFilePath, zero2neg) 
- y: the y vector generated by parseCSV(csvFilePath, zero2neg) 
- r: the learning rate to use
- T: the number of epochs to run the algorithm for

Outputs:

- a: the learned averaged weight vector


ppredict_AveragedPerceptron(x, a) predicts the labels for each row of x using the weight vector a.

Inputs:

- x: the example list genereated from parseCSV_NoLabel(csvFilePath) or parseCSV(csvFilePath, zero2neg)
- a: the learned averaged weight vector to be used to predict the examples

Outputs:

- predictions: a numpy vector where each element is the predicted label in {-1, 1} for the corresponding example

An example of how to use AveragedPerceptron can be found in the file Perceptron/hw3_perceptron.py
