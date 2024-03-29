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


predict_AveragedPerceptron(x, a) predicts the labels for each row of x using the weight vector a.

Inputs:

- x: the example list genereated from parseCSV_NoLabel(csvFilePath) or parseCSV(csvFilePath, zero2neg)
- a: the learned averaged weight vector to be used to predict the examples

Outputs:

- predictions: a numpy vector where each element is the predicted label in {-1, 1} for the corresponding example

An example of how to use AveragedPerceptron can be found in the file Perceptron/hw3_perceptron.py





## Support Vector Machines (SVM)

To obtain the properly formatted example list for all SVM algorithms, the function SVM.parseCSV(csvFilePath, zero2neg) in SVM.py should be called.

SVM.parseCSV(csvFilePath, zero2neg) creates the correct x matrix and y vector from the provided csv file.

Inputs:

- csvFilePath: a string that is the path to a .csv file containing the example data with no headers and delimited by a comma. All values in the csv should be real numbers.
- zero2neg: A boolean value. True indicates that the label column is in the range {0, 1} which must be converted to the range {-1, 1}. A False value will skip the conversion.

Outputs:

- x: a numpy matrix where each row is an example's attribute value as a float
- y: a numpy vector where each element is the corresponding row in x's label value as a float

For csv files containg data with no label column, such as test data, parseCSV_NoLabel(csvFilePath) is used to generate just the proper example matrix x.


SVM.GammaSchedule dataclass provides a way to use learning rate schedules. It has three data fields:

- schedule: The schedule to use. 1 will calculate iteration gamma as gamma = GammaSchedule.gamma0 / (1 + (GammaSchedule.gamma0 * iterations / GammaSchedule.d)). 2 will use the formula gamma = GammaSchedule.gamma0 / (1 + iterations)
- gamma0: The gamma0 used to calculate the iteration gamma
- d: The d used to calculate the iteration gamma, ignored when schedule is 2



## SVM in Primal Domain With Stochastic Sub-gradient Descent

The implementation of SVM in the primal domain with stochastic sub-gradient descent is found in the SVM directory. SVM.py contains the implementation.

To obtain properly formatted x and y, the function SVM.parseCSV(csvFilePath, zero2neg) is called as described above.

SVM_primalSGD(x, y, GammaSchedule, C, T, retCostList) implements the learning algorithm.

Inputs:

- x: the x matrix generated by parseCSV(csvFilePath, zero2neg) 
- y: the y vector generated by parseCSV(csvFilePath, zero2neg)
- GammaSchedule: An SVM.GammaSchedule dataclass which has three data fields: schedule, gamma0, and d set as described above
- C: The SVM hyperparamter C that balances the margin and number of mistakes. A larger C will favor smaller margin separators with less mistakes while a smaller C will favor larger margin separators that allow more mistakes.
- T: the number of epochs to run the algorithm for
- retCostList: dictates whether to return the value of the cost function after each iteration. If True, a list of cost function values will be returned where the index is the iteration the value was calculated on. If False, the list will be empty

Outputs:

- wghts: The learned weight vector that can be used for prediction
- j_list: If retCostList was True, this will be a list of cost function values at each iteration. If retCostList was false, this will be an empty list


predict_SVM_primalSGD(x, w) predicts the labels for each row of x using weight vector w

Inputs:

- x: the example matrix genereated from parseCSV_NoLabel(csvFilePath) or parseCSV(csvFilePath, zero2neg)
- w: the learned weight vector to be used to predict the examples

Outputs:

- predictions: a numpy vector where each element is the predicted label in {-1, 1} for the corresponding example

An example of how to use SVM_primalSGD can be found in the file SVM/hw4_svm.py




## SVM in Dual Domain

The implementation of SVM in the dual domain is found in the SVM directory. SVM.py contains the implementation.

To obtain properly formatted x and y, the function SVM.parseCSV(csvFilePath, zero2neg) is called as described above.

SVM_dual(x, y, C) implements the learning algorithm.

Inputs:

- x: the x matrix generated by parseCSV(csvFilePath, zero2neg) 
- y: the y vector generated by parseCSV(csvFilePath, zero2neg)
- C: The SVM hyperparamter C that balances the margin and number of mistakes. A larger C will favor smaller margin separators with less mistakes while a smaller C will favor larger margin separators that allow more mistakes.

Outputs:

- w: The learned weight vector that can be used for prediction
- b: The bias parameter of the weight vecotr to be used for prediction


predict_SVM_dual(x, w, b) predicts the labels for each row of x using weight vector w

Inputs:

- x: the example matrix genereated from parseCSV_NoLabel(csvFilePath) or parseCSV(csvFilePath, zero2neg)
- w: the learned weight vector to be used to predict the examples
- b: the learned bias paramater to be used to predict the examples

Outputs:

- predictions: a numpy vector where each element is the predicted label in {-1, 1} for the corresponding example

An example of how to use SVM_dual can be found in the file SVM/hw4_svm.py





## SVM in Dual Domain with Gaussian Kernel

The implementation of SVM in the dual domain with the Gaussian kernel is found in the SVM directory. SVM.py contains the implementation.

To obtain properly formatted x and y, the function SVM.parseCSV(csvFilePath, zero2neg) is called as described above.

SVM_dualKernelGaussian(x, y, C, g) implements the learning algorithm.

Inputs:

- x: the x matrix generated by parseCSV(csvFilePath, zero2neg) 
- y: the y vector generated by parseCSV(csvFilePath, zero2neg)
- C: The SVM hyperparamter C that balances the margin and number of mistakes. A larger C will favor smaller margin separators with less mistakes while a smaller C will favor larger margin separators that allow more mistakes.
- g: the gamma paramter in the Gaussian kernel

Outputs:

- a: The vector of optimal alpha values that can be used for prediction


predict_SVM_dualKernelGaussian(x, a, x_train, y, g) predicts the labels for each row of x using optimal vector a and the training examples

Inputs:

- x: the example matrix genereated from parseCSV_NoLabel(csvFilePath) or parseCSV(csvFilePath, zero2neg)
- a: the learned optimial vector to be used to predict the examples
- x_train: the example matrix used during training of a
- y: the label vector of the training examples in x_train
- g: the gamma paramter for the Gaussian kernel used during training

Outputs:

- predictions: a numpy vector where each element is the predicted label in {-1, 1} for the corresponding example

An example of how to use SVM_dualKernelGaussian can be found in the file SVM/hw4_svm.py





## Kernel Perceptron with Gaussian Kernel

The implementation of kernel Perceptron with the Gaussian kernel is found in the SVM directory. SVM.py contains the implementation.

To obtain properly formatted x and y, the function SVM.parseCSV(csvFilePath, zero2neg) is called as described above.

Perceptron_Kernel_Gaussian(x, y, g, T) implements the learning algorithm.

Inputs:

- x: the x matrix generated by parseCSV(csvFilePath, zero2neg) 
- y: the y vector generated by parseCSV(csvFilePath, zero2neg)
- g: the gamma paramter in the Gaussian kernel
- T: the number of epochs to run the algorithm for

Outputs:

- c: The vector of mistake counts that can be used for prediction


predict_Perceptron_Kernel_Gaussian(x, c, x_train, y, g) predicts the labels for each row of x using mistake  count vector c and the training examples

Inputs:

- x: the example matrix genereated from parseCSV_NoLabel(csvFilePath) or parseCSV(csvFilePath, zero2neg)
- c: the learned mistake count vector to be used to predict the examples
- x_train: the example matrix used during training of c
- y: the label vector of the training examples in x_train
- g: the gamma paramter for the Gaussian kernel used during training

Outputs:

- predictions: a numpy vector where each element is the predicted label in {-1, 1} for the corresponding example

An example of how to use Perceptron_Kernel_Gaussian can be found in the file SVM/hw4_svm.py






## Logistic Regression

To obtain the properly formatted example list for all Logistic Regression algorithms, the function LogisticRegression.parseCSV(csvFilePath, zero2neg) in LogisticRegression.py should be called.

LogisticRegression.parseCSV(csvFilePath, zero2neg) creates the correct x matrix and y vector from the provided csv file.

Inputs:

- csvFilePath: a string that is the path to a .csv file containing the example data with no headers and delimited by a comma. All values in the csv should be real numbers.
- zero2neg: A boolean value. True indicates that the label column is in the range {0, 1} which must be converted to the range {-1, 1}. A False value will skip the conversion.

Outputs:

- x: a numpy matrix where each row is an example's attribute value as a float
- y: a numpy vector where each element is the corresponding row in x's label value as a float


LogisticRegression.GammaSchedule dataclass provides a way to use learning rate schedules. Both logistic regression algorithm use the same learning rate schedule: gamma_iteration = GammaSchedule.gamma0 / (1 + (GammaSchedule.gamma0 * iterations / GammaSchedule.d)). It has two data fields:

- gamma0: The gamma0 used to calculate the iteration gamma
- d: The d used to calculate the iteration gamma


## Logistic Regression with stochastic gradient descent and MAP estimation

The implementation of this Logistic Regression algorithm is found in the LogisticRegression directory. LogisticRegression.py contains the implementation.

To obtain properly formatted x and y, the function LogisticRegression.parseCSV(csvFilePath, zero2neg) is called as described above.

LogisticRegression_SGD_MAP(x, y, T, v, GammaSchedule, checkConverge) implements the learning algorithm.

Inputs:

- x: the x matrix generated by parseCSV(csvFilePath, zero2neg) 
- y: the y vector generated by parseCSV(csvFilePath, zero2neg)
- T: the number of epochs to run the algorithm for
- v: the prior variance hyperpramater for the Gaussian prior distribution
- GammaScedule: the GammaSchedule that contians the gamma0 and d for the learning rate schedule
- checkConverge: when true, a list of objective function values at each epoch is returned. This can be plotted to check for convergence. When false, the returned list is empty

Outputs:

- wghts: The learned weight vector that can be used for prediction
- lossList: if checkConverge is true, a list of objective function values at each epoch; if false, an empty list


LogisticRegression_SGD_MAP_predict(x, w) predicts the labels for each row of x using the learned weight vector w

Inputs:

- x: the example matrix genereated from parseCSV_NoLabel(csvFilePath) or parseCSV(csvFilePath, zero2neg)
- w: the learned weight vector from LogisticRegression_SGD_MAP()

Outputs:

- predictions: a numpy vector where each element is the predicted label in {-1, 1} for the corresponding example

An example of how to use LogisticRegression_SGD_MAP can be found in the file LogisticRegression/hw5_lr.py



## Logistic Regression with stochastic gradient descent and ML estimation

The implementation of this Logistic Regression algorithm is found in the LogisticRegression directory. LogisticRegression.py contains the implementation.

To obtain properly formatted x and y, the function LogisticRegression.parseCSV(csvFilePath, zero2neg) is called as described above.

LogisticRegression_SGD_ML(x, y, T, GammaSchedule, checkConverge) implements the learning algorithm.

Inputs:

- x: the x matrix generated by parseCSV(csvFilePath, zero2neg) 
- y: the y vector generated by parseCSV(csvFilePath, zero2neg)
- T: the number of epochs to run the algorithm for
- GammaScedule: the GammaSchedule that contians the gamma0 and d for the learning rate schedule
- checkConverge: when true, a list of objective function values at each epoch is returned. This can be plotted to check for convergence. When false, the returned list is empty

Outputs:

- wghts: The learned weight vector that can be used for prediction
- lossList: if checkConverge is true, a list of objective function values at each epoch; if false, an empty list


LogisticRegression_SGD_ML_predict(x, w) predicts the labels for each row of x using the learned weight vector w

Inputs:

- x: the example matrix genereated from parseCSV_NoLabel(csvFilePath) or parseCSV(csvFilePath, zero2neg)
- w: the learned weight vector from LogisticRegression_SGD_ML()

Outputs:

- predictions: a numpy vector where each element is the predicted label in {-1, 1} for the corresponding example

An example of how to use LogisticRegression_SGD_ML can be found in the file LogisticRegression/hw5_lr.py




## Neural Networks

To obtain the properly formatted example list for the Neural Network algorithm, the function NeuralNetwork.parseCSV(csvFilePath, zero2neg) in NeuralNetwork.py should be called.

NeuralNetwork.parseCSV(csvFilePath, zero2neg) creates the correct x matrix and y vector from the provided csv file.

Inputs:

- csvFilePath: a string that is the path to a .csv file containing the example data with no headers and delimited by a comma. All values in the csv should be real numbers.
- zero2neg: A boolean value. True indicates that the label column is in the range {0, 1} which must be converted to the range {-1, 1}. A False value will skip the conversion.

Outputs:

- x: a numpy matrix where each row is an example's attribute value as a float
- y: a numpy vector where each element is the corresponding row in x's label value as a float


NeuralNetwork.GammaSchedule dataclass provides a way to use learning rate schedules. Both logistic regression algorithm use the same learning rate schedule: gamma_iteration = GammaSchedule.gamma0 / (1 + (GammaSchedule.gamma0 * iterations / GammaSchedule.d)). It has two data fields:

- gamma0: The gamma0 used to calculate the iteration gamma
- d: The d used to calculate the iteration gamma


A NeuralNetwork.NeuralNet object is needed for the algorithms. It is initialized using:

NeuralNetwork.NeuralNet(layers, numInputs, hiddenNodeCount, randInit) where

- layers: the number of layers in the neural network
- numInputs: the number of inputs into the neural network, including an augmented 1
- hiddenNodeCount: a list containing the number of hidden nodes at each layer; hiddenNodeCount[0] corresponds to the first hidden layer
- randInit: if True, the weights are randomly intialized using the standard Gaussion distribution; if false, the weights are intialized to 0



## Neural Network with stochastic gradient descent

The implementation of this Neural Network with stochastic gradient descent algorithm is found in the NeuralNetworks directory. NeuralNetwork.py contains the implementation.

To obtain properly formatted x and y, the function NeuralNetwork.parseCSV(csvFilePath, zero2neg) is called as described above.

NeuralNetwork_SGD(x, y, nn, GammaSchedule, T, checkConverge) implements the learning algorithm assuming an architecture described in the homework 5 handout: the first node in each hidden layer and input is an augmented 1, there is one output, and every node is connected to every node in the layer above excpet the augmented 1. The sigmoid activation function is used along with squared loss for the output.

Inputs:

- x: the x matrix generated by parseCSV(csvFilePath, zero2neg) 
- y: the y vector generated by parseCSV(csvFilePath, zero2neg)
- nn: a NeuralNet object created as described above
- GammaScedule: the GammaSchedule that contians the gamma0 and d for the learning rate schedule
- T: the number of epochs to run the algorithm for
- checkConverge: when true, a list of objective function values at each epoch is returned. This can be plotted to check for convergence. When false, the returned list is empty

Outputs:

- nn: a copy of the NeuralNet input with final weights in the nn.weights matrix
- lossList: if checkConverge is true, a list of objective function values at each epoch; if false, an empty list


NeuralNetwork_SGD_predict(x, nn) predicts the labels for each row of x using the NeuralNet nn

Inputs:

- x: the example matrix genereated from parseCSV_NoLabel(csvFilePath) or parseCSV(csvFilePath, zero2neg)
- nn: the NeuralNet returned by NeuralNetwork_SGD()

Outputs:

- predictions: a numpy vector where each element is the predicted label in {-1, 1} for the corresponding example

An example of how to use NeuralNetwork_SGD can be found in the file NeuralNetwork/hw5_nn.py
