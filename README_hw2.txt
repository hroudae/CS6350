To run the program to produce the results specified in homework 2, there
are two shell scripts: EnsembleLearning/run.sh and 
LinearRegression/run.sh. These will call the five python scripts 
that produce the specified data. They must be run while inside their
respective directories. 

EnsembleLearning/run.sh runs the folowing python scripts:

EnsembleLearning/hw2_adaboost.py:
This script runs the AdaBoost learning algorithm for the bank datasets and
outputs the traing and test errors per iteration to bank_errors_adaboost.csv
and the training and test errors of each decision stump to
bank_stumperrors_adaboost.csv.


EnsembleLearning/hw2_baggedtrees.py:
This script runs the Bagged Decison Trees learning algorithm for the bank
dataset and outputs the per iteration training and test errors to 
bank_errors_bagged.csv. It also performs the bias, variance, and squared
error calculations for a single tree and the bagged trees which is printed
to the screen.


EnsembleLearning/hw2_randomforests.py:
This script runs the Random Forests learning algorithm fo the bank dataset
and outputs the training and test errors per iteration to three files,
depending on the feature subset size: bank_errors_randforests_featsz{sz}.csv
where {sz} is either 2, 4, or 6. It also performs the bias, variance, and squared
error calculations for a single tree and the bagged trees which is printed
to the screen.


EnsembleLearning/hw2_credit.py
This script runs the Bagged Decision Trees, Random Forests, and Adaboost
learning algorithm for the credit dataset. The commented out code will
randomly select 24000 exampes to be used for training with the remaining
6000 for testing and then writes them to credit/train.csv and credit/test.csv
so the same random sample can be used again if the script is interrupted.
The per iteration training and test errors are written to the following files:
credit_errors_bagged.csv
credit_errors_randforests_featsz{sz}.csv
credit_errors_adaboost.csv
with similar naming conventions as the bank dataset errors. The training and
test error for a single full expanded decision tree is printed to the screen.


LinearRegression/run.sh runs the following python script:

LinearRegression/hw2_gradients.py
This script runs batch gradient descent learning algorithm on the concrete
dataset and outputs the per iteration cost function value of the training 
dataset to concrete_costvals_bgd.csv and outputs the test data cost function
value using the learned weight vector to the screen. It also runs stochastic
gradient descent algorithm on the same dataset and outputs the per iteration
cost function value of the training dataset to concrete_costvals_sgd.csv as
well as outputs test data cost function value using the learned weight vector
to the screen. Finally, it calculates the optimal weight vector using the
analytical form and outputs the result as well as the test data cost function 
value using the weight vector to the screen.
