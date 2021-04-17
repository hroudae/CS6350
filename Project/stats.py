##########
# Author: Evan Hrouda
# Purpose: Create various graphs to facilitate in processing the data
##########
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from pandas import DataFrame, read_csv, get_dummies
import pandas as pd
import seaborn as sns
from matplotlib import rcParams
rcParams.update({'figure.autolayout': True})

train_data = "data/train_final.csv"
df = read_csv(train_data)

# sum capital gains into one
df['capital'] = df[['capital.gain', 'capital.loss']].sum(axis=1)


# Plot and save a correlation matrix
num_feat = df.select_dtypes(include=['float', 'int']).columns
sns.set(style="white")

# Compute the correlation matrix
corr = df[num_feat].corr()
# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(8, 5))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap=True)
plt.tight_layout()
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr, cmap=cmap, center=0, vmax=0.3,
            annot=True)
f.savefig("correlMatrix.png")



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


# Some bar graphs
plt.figure(figsize=(18,25))
plt.subplot(521)

i=0
for cats in df.select_dtypes(include=['object', 'category']).columns:
    plt.subplot(5, 3, i+1)
    i += 1
    # sns.countplot(x='income>50K', data=df, hue=cats)
    sns.countplot(y=cats, data=df, hue='income>50K')
    plt.title(cats)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.subplot(5, 3, i+1)
df_noNA = df[df.region != 'NA']
sns.countplot(y='region', data=df_noNA, hue='income>50K')
plt.title('Region with no NA')
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

plt.savefig("stats.png")
