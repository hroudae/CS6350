##########
# Author: Evan Hrouda
# Purpose: Try various ML algorithms to predict whether someone's
#          income is >50k. Utilizes the GridSearchCV function which
#          performs an exhaustive search over a paramater grid and 
#          5-fold cross validaiton in order to find the best hyperparameters
##########
import matplotlib
from pandas.core.arrays import categorical
from pandas.core.arrays.sparse import dtype
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame, read_csv, get_dummies
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn import svm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})



#####
# Author: Evan Hrouda
# Purpose: prepare the data in the dataframe for the evaluation
#####
def prepareData(df, train):
    df['capital'] = df['capital.gain'] - df['capital.loss']
    # convert numerical to categorical
    age_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, np.inf]
    age_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12']
    df['agerange'] = pd.cut(df['age'], age_bins, labels=age_names)

    hours_bins = [0, 36, 40, np.inf]
    hours_names = ['Part time', 'Normal', 'Overtime']
    df['hoursrange'] = pd.cut(df['hours.per.week'], hours_bins, labels=hours_names)

    cap_bins = [np.NINF, 0, 5000, np.inf]
    cap_names = ['<0', '0 to 5000', '> 5000']
    df['capitalrange'] = pd.cut(df['capital'], cap_bins, labels=cap_names)

    # Combine some categories
    relat_mapping = {
                'Husband':'Spouse',
                'Wife':'Spouse',
                'Own-child':'Child',
                'Not-in-family':'Other',
                'Other-relative':'Other', 
                'Unmarried':'Other'
               }
    df['relationship'] = df['relationship'].map(relat_mapping)

    marry_mapping = {
                'Married-civ-spouse': 'Married',
                'Divorced': 'Post-marriage',
                'Never-married': 'Never-married',
                'Separated': 'Post-marriage',
                'Widowed': 'Post-marriage',
                'Married-spouse-absent': 'Post-marriage',
                'Married-AF-spouse': 'Married'
               }
    df['marital.status'] = df['marital.status'].map(marry_mapping)


    country_mapping = {
        'United-States': 'NA',
        'Cambodia': 'Asia',
        'England': 'EUR',
        'Puerto-Rico': 'Caribbean',
        'Canada': 'NA',
        'Germany': 'EUR',
        'Outlying-US(Guam-USVI-etc)': 'Caribbean',
        'India': "Asia",
        'Japan': 'Asia',
        'Greece': 'EUR',
        'South': 'Other',
        'China': 'Asia',
        'Cuba': 'Caribbean',
        'Iran': 'ME',
        'Honduras': 'CA',
        'Philippines': 'Asia',
        'Italy': 'EUR',
        'Poland': 'EUR',
        'Jamaica': 'Caribbean',
        'Vietnam': 'Asia',
        'Mexico': 'CA',
        'Portugal': 'EUR',
        'Ireland': 'EUR',
        'France': 'EUR',
        'Dominican-Republic': 'Caribbean',
        'Laos': 'Asia',
        'Ecuador': 'SA',
        'Taiwan': 'Asia',
        'Haiti': 'Caribbean',
        'Columbia': 'SA',
        'Hungary': 'EUR',
        'Guatemala': 'CA',
        'Nicaragua': 'CA',
        'Scotland': 'EUR',
        'Thailand': 'Asia',
        'Yugoslavia': 'EUR',
        'El-Salvador': 'CA',
        'Trinadad&Tobago': 'Caribbean',
        'Peru': 'SA',
        'Hong': 'Asia',
        'Holand-Netherlands': 'EUR',
        '?': 'Other'
    }
    df['region'] = df['native.country'].map(country_mapping)

    # sex to binary
    sex_mapping = {
               'Male': 1,
               'Female': 0}
    df['sex'] = df['sex'].map(sex_mapping)
    # change {0, 1} to {-1, 1}
    if train == True:
        income_mapping = {
                   1: 1,
                   0: -1}
        df['income>50K'] = df['income>50K'].map(income_mapping)

    #delete the now redundant columns
    del df['age'], df['native.country'], df['capital'], df['capital.gain'], df['capital.loss'], df['hours.per.week']
    return df

def getOrdinalAndOnehot(df, combo):
    # onehot = get_dummies(df, dtype=int) # one hot encoding for numerical algorithms
    onehot = pd.DataFrame(preprocessing.OneHotEncoder(sparse=False, dtype=np.int64).fit(combo).transform(df))
    ordin = pd.DataFrame(preprocessing.OrdinalEncoder(dtype=np.int64).fit(combo).transform(df))
    
    return onehot, ordin



# read in the data
train_data = "data/train_final.csv"
df = read_csv(train_data)
df_test = read_csv("data/test_final.csv")

# fnlwgt has almost 0 correlation with income, so delete it
del df['fnlwgt'], df_test['fnlwgt']
del df_test["ID"] # get rid of ID column

df_noprep = df.copy()
df_test_noprep = df_test.copy()

# sex to binary
sex_mapping = {
            'Male': 1,
            'Female': 0}
df_noprep['sex'] = df_noprep['sex'].map(sex_mapping)
df_test_noprep['sex'] = df_test_noprep['sex'].map(sex_mapping)
# change {0, 1} to {-1, 1}
income_mapping = {
                1: 1,
               0: -1}
df_noprep['income>50K'] = df_noprep['income>50K'].map(income_mapping)

df = prepareData(df, True)
df_test = prepareData(df_test, False)

y = df["income>50K"]
del df["income>50K"], df_noprep["income>50K"]

# make sure all attribute values are accounted for in the encodingd
combo = pd.concat([df, df_test])
combo2 = pd.concat([df_noprep, df_test_noprep])

onehot, ordin = getOrdinalAndOnehot(df, combo)
onehot_test, ordin_test = getOrdinalAndOnehot(df_test, combo)

onehot_noprep, ordin_noprep = getOrdinalAndOnehot(df_noprep, combo2)
onehot_test_noprep, ordin_test_noprep = getOrdinalAndOnehot(df_test_noprep, combo2)



# Rename some data for easier readability
# base data
x = df
x_test = df_test

# one-hot data
y_one = y
x_one = onehot
x_one_noprep = onehot_noprep

# ordinal data
y_ordin = y
x_ordin = ordin
x_ordin_noprep = ordin_noprep


# Setup a grid search
# Grid search exhaustively searches from a grid of parameters to find the highest 
# cross validation score. The estimator with highest cross-validation score is
# in the attribute best_estimator_. It deafuaults to 5-fold cross validation


# # adaboost
# parameters = {"base_estimator__criterion" : ["gini", "entropy"],
#               "base_estimator__max_depth" : [1],
#               "base_estimator__splitter" :   ["best", "random"],
#               "base_estimator__random_state" : [1, 60, 101],
#               'n_estimators':[500, 1000, 5000],
#               'algorithm':('SAMME', 'SAMME.R')
#              }
# parameters_one = parameters
# DTC = DecisionTreeClassifier(max_features = "auto", class_weight = "balanced")
# clf = AdaBoostClassifier(base_estimator = DTC)
# with open("best_parameters.txt",'a') as out:
#     out.write("\n\n\n\nAdaboost\n")



# SVM
# parameters = {"C" : [0.001, 0.01, 0.1, 1, 10, 100],
#             'kernel' : ['linear', 'poly', 'rbf', 'sigmoid'],
#             'gamma' : ['scale', 'auto']
#              }
# parameters_one = parameters
# clf = svm.SVC()
# with open("best_parameters.txt",'a') as out:
#     out.write("\n\n\n\nSVM\n")




# Logisitic Regression
parameters = {"penalty" : ['l1', 'l2', 'elasticnet', 'none'],
            "C" : [0.001, 0.01, 0.1, 1, 10, 100],
            'fit_intercept' : [True, False],
            'solver' : ['newton-cg', 'lbfgs', 'sag', 'saga'],
            'max_iter' : [100, 200, 300]
             }
parameters_one = parameters
clf = LogisticRegression()
with open("best_parameters.txt",'a') as out:
    out.write("\n\n\n\nLogisitic Regression\n")


# Random Forest
# parameters = {
#     "n_estimators" : [50, 100, 250, 500],
#     "criterion" : ['gini', 'entropy'],
#     "max_features" : ['auto', 'sqrt', 'log2']
# }
# parameters_one = parameters
# clf = RandomForestClassifier(n_jobs=-1)
# with open("best_parameters.txt",'a') as out:
#     out.write("\n\n\n\nRandom Forests\n")



# SGD
# parameters = {
#     "loss" : ['hinge', 'log', 'squared_hinge', 'perceptron'],
#     "penalty" : ['l1', 'l2', 'elasticnet'],
#     "alpha" : [0.0001, 0.01, 1, 10],
#     'max_iter' : [500, 1000, 5000],
#     'learning_rate' : ['constant', 'optimal', 'invscaling', 'adaptive'],
#     'eta0' : [0.001, 0.01, 0.1, 1, 10],
#     'power_t' : [0.25, 0.5, 0.75]
# }
# parameters_one = parameters
# clf = SGDClassifier(n_jobs=-1)
# with open("best_parameters.txt",'a') as out:
#     out.write("\n\n\n\nSGD\n")



# Perceptron
# parameters = {
#     'penalty' : ['l1', 'l2', 'elasticnet'],
#     'alpha' : [0.0001, 0.01, 1, 10],
#     'fit_intercept' : [True, False],
#     'max_iter' : [500, 1000, 5000],
#     'eta0' : [0.001, 0.01, 0.1, 1, 10]
# }
# parameters_one = parameters
# clf = Perceptron(n_jobs=-1)
# with open("best_parameters.txt",'a') as out:
#     out.write("\n\n\n\nPerceptron\n")


# Decision Trees
# parameters = {
#     'criterion' : ['gini', 'entropy'],
#     "splitter" :   ["best", "random"],
#     'max_depth' : [None, 1, 2, 5, 10]
# }
# parameters_one = parameters
# clf = DecisionTreeClassifier()
# with open("best_parameters.txt",'a') as out:
#     out.write("\n\n\n\nDecisionTrees\n")


# ANN - Multi-layer Perceptron - search hidden_layer_sizes??
# parameters = {
#     'activation' : ['tanh', 'relu'],
#     'solver' : ['lbfgs', 'sgd', 'adam'],
#     'alpha' : [0.0001, 0.01, 1],
#     'max_iter' : [200, 500],
#     'learning_rate' : ['constant', 'invscaling', 'adaptive'],
#     'learning_rate_init' : [0.001, 0.01, 0.1, 1]
# }
# parameters_one = parameters
# clf = MLPClassifier()
# with open("best_parameters.txt",'a') as out:
#     out.write("\n\n\n\nNN_MLP\n")



print("One hot")
grid_one = GridSearchCV(clf, parameters_one, n_jobs=-1, scoring='roc_auc')
grid_one.fit(x_one, y_one)
print(grid_one.best_params_)
b = grid_one.best_estimator_
p = b.predict(x_one)
print(f"Train Mean accuracy: {b.score(x_one, y_one)}")
train_auc = roc_auc_score(np.array(y_one), b.decision_function(x_one))
# for Random Forest and Decision Tree
# train_auc = roc_auc_score(np.array(y_one), b.predict_proba(x_one)[:,1])
print(f"Train AUC: {train_auc}")
numWrong = sum(abs(p-y_one) / 2)
train_err = numWrong/len(y_one)
print(f"Train Error: {train_err}")

with open("best_parameters.txt",'a') as out:
    out.write(f"Prep One-hot: train: {train_err}\tauc train: {train_auc}\n")
    out.write(f"{grid_one.best_params_}\n")




print()
print("No prep")

print("One hot")
grid_one = GridSearchCV(clf, parameters_one, n_jobs=-1, scoring='roc_auc')
grid_one.fit(x_one_noprep, y_one)
print(grid_one.best_params_)
b = grid_one.best_estimator_
p = b.predict(x_one_noprep)
print(f"Train Mean accuracy: {b.score(x_one_noprep, y_one)}")
train_auc = roc_auc_score(np.array(y_one), b.decision_function(x_one_noprep))
# for Random Forest and Decision Tree
# train_auc = roc_auc_score(np.array(y_one), b.predict_proba(x_one_noprep)[:,1])
print(f"Train AUC: {train_auc}")
numWrong = sum(abs(p-y_one) / 2)
train_err = numWrong/len(y_one)
print(f"Train Error: {train_err}")

with open("best_parameters.txt",'a') as out:
    out.write(f"NoPrep One-hot: train: {train_err}\tauc train: {train_auc}\n")
    out.write(f"{grid_one.best_params_}\n")






print()
print("Ordinal")
grid = GridSearchCV(clf, parameters, n_jobs=-1, scoring='roc_auc')
grid.fit(x_ordin, y_ordin)
print(grid.best_params_)
b = grid.best_estimator_
p = b.predict(x_ordin)
print(f"Train Mean accuracy: {b.score(x_ordin, y_ordin)}")
train_auc = roc_auc_score(np.array(y_ordin), b.decision_function(x_ordin))
# for Random Forest and Decision Tree
# train_auc = roc_auc_score(np.array(y_ordin), b.predict_proba(x_ordin)[:,1])
print(f"Train AUC: {train_auc}")
numWrong = sum(abs(p-y_ordin) / 2)
train_err = numWrong/len(y_ordin)
print(f"Train Error: {train_err}")

with open("best_parameters.txt",'a') as out:
    out.write(f"Prep Ord: train: {train_err}\tauc train: {train_auc}\n")
    out.write(f"{grid.best_params_}\n")


print()
print("No prep")

print()
print("Ordinal")
grid = GridSearchCV(clf, parameters, n_jobs=-1, scoring='roc_auc')
grid.fit(x_ordin_noprep, y_ordin)
print(grid.best_params_)
b = grid.best_estimator_
p = b.predict(x_ordin_noprep)
print(f"Train Mean accuracy: {b.score(x_ordin_noprep, y_ordin)}")
train_auc = roc_auc_score(np.array(y_ordin), b.decision_function(x_ordin_noprep))
# for Random Forest and Decision Tree
# train_auc = roc_auc_score(np.array(y_ordin), b.predict_proba(x_ordin_noprep)[:,1])
print(f"Train AUC: {train_auc}")
numWrong = sum(abs(p-y_ordin) / 2)
train_err = numWrong/len(y_ordin)
print(f"Train Error: {train_err}")

with open("best_parameters.txt",'a') as out:
    out.write(f"NoPrep Ord: train: {train_err}\tauc train: {train_auc}\n")
    out.write(f"{grid.best_params_}\n")

