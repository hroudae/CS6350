##########
# Author: Evan Hrouda
# Purpose: Write predictions using the parameters found from GridSearchCV
#          which is implemented in gridsearch.py
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

combo = pd.concat([df, df_test])
combo2 = pd.concat([df_noprep, df_test_noprep])

onehot, ordin = getOrdinalAndOnehot(df, combo)
onehot_test, ordin_test = getOrdinalAndOnehot(df_test, combo)

onehot_noprep, ordin_noprep = getOrdinalAndOnehot(df_noprep, combo2)
onehot_test_noprep, ordin_test_noprep = getOrdinalAndOnehot(df_test_noprep, combo2)


# The various algorithms and parameters found
# uncomment the one to be run

# params = {'C': 5, 'gamma': 'scale', 'kernel': 'linear'}
# clf = svm.SVC()

# params = {'C': 0.001, 'fit_intercept': False, 'max_iter': 300, 'penalty': 'none', 'solver': 'lbfgs'}
# clf = LogisticRegression()

# params = {'algorithm': 'SAMME.R', 'base_estimator__criterion': 'gini', 'base_estimator__max_depth': 1, 'base_estimator__random_state': 1, 'base_estimator__splitter': 'best', 'n_estimators': 5000}
# DTC = DecisionTreeClassifier(max_features = "auto", class_weight = "balanced")
# clf = AdaBoostClassifier(base_estimator = DTC)

# params = {'criterion': 'gini', 'max_features': 'sqrt', 'n_estimators': 500}
# clf = RandomForestClassifier(n_jobs=-1)

# params = {'alpha': 0.0001, 'eta0': 0.01, 'learning_rate': 'adaptive', 'loss': 'squared_hinge', 'max_iter': 5000, 'penalty': 'l2', 'power_t': 0.25}
# clf = SGDClassifier(n_jobs=-1)

# params = {'alpha': 0.0001, 'eta0': 0.001, 'fit_intercept': True, 'max_iter': 500, 'penalty': 'l2'}
# clf = Perceptron(n_jobs=-1)

params = {'criterion': 'entropy', 'max_depth': 10, 'splitter': 'best'}
clf = DecisionTreeClassifier()

# uncomment the dataset to use
# # One-hot, preprocess
# x = onehot
# x_test = onehot_test
# # One-hot, no preprocess
# x = onehot_noprep
# x_test = onehot_test_noprep
# # Ordinal, preprocess
# x = ordin
# x_test = ordin_test
# Ordinal, no preprocess
x = ordin_noprep
x_test = ordin_test_noprep

clf.set_params(**params)
clf.fit(x, y)
p = clf.predict(x)
print(f"Train Mean accuracy: {clf.score(x, y)}")
train_auc = roc_auc_score(np.array(y), clf.predict_proba(x)[:,1]) # for Random Forest, Dec Tree
# train_auc = roc_auc_score(np.array(y), clf.decision_function(x))
print(f"Train AUC: {train_auc}")
numWrong = sum(abs(p-y) / 2)
train_err = numWrong/len(y)
print(f"Train Error: {train_err}")

#Write the predictions to a file
# p = clf.decision_function(x_test)
p = clf.predict_proba(x_test)[:,1] # for Random Forest, Dec Tree
with open("predicts.csv",'w') as out:
    out.write("ID,Prediction\n")
    for i in range(len(p)):
        out.write(f"{i+1},{p[i]}\n")

print("Predictions written to predicts.csv")

