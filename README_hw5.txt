To run the program to produce the results specified in homework 5, the
shell script run.sh should be run. This will call the two python scripts 
that produce the specified data. It will first cd into the LogisticRegression
directory and run the python script hw5_lr.py to produce to results for the
logistic regression problems. It will then cd into the NeuralNetworks directory
and run the python script hw5_nn.py which produces the results for the 
neural network problems. All results are printed to the screen. Both scripts
have checkConverge booleans which, when True, will plot the loss function over
epochs in order to check for convergence.

The implementations of logistic regression stochastic gradient descent with
MAP and ML estimations are found in the file LogisticRegression/LogisticRegression.py

The implementation of the nerual network is found in the file NeuralNetworks/NeuralNetwork.py

