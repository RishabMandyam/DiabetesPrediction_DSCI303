import pandas as pd
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import numpy as np

df = pd.read_csv('diabetes_012_health_indicators.csv')
y = df.Diabetes_012

# calculate the missing ratio of each features column in dataframe
df_na = (df.isnull().sum() / len(df)) * 100
df_na = df_na.drop(df_na[df_na == 0].index).sort_values(ascending=False)

# Visualize the missing ratios
f, ax = plt.subplots(figsize=(10, 8))
plt.xticks(rotation=90)
sns.barplot(x=df_na.index, y=df_na)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by features', fontsize=15)

## NO missing data!

print(df['BMI'].describe())
plt.figure(figsize=(9, 8))

# Plot the distribution of blood pressure
sns.displot(df['BMI'])

sns.displot((df['Diabetes_012']))

# Select the numerical featrues
df_num = df.select_dtypes(include = 'number')
df_num

# Plot the distribution of each numeric featrues [hint: you can use pandas: df.hist()]
df.hist(df_num.columns, figsize=(16, 20), bins=50, xlabelsize=8, ylabelsize=8)

df_cate = df.select_dtypes(include = ['O'])
print('There is {} non numerical features including:\n{}'.format(len(df_cate.columns), df_cate.columns.tolist()))

# Select only numeric features before calculating correlation
df_corr = df.select_dtypes(include=['number']).corr()
plt.figure(figsize=(11, 11))

sns.heatmap(df_corr,
            cmap='viridis', vmax=1.0, vmin=-1.0, linewidths=0.1,
            annot=True, annot_kws={"size": 8}, square=True);

print(df_corr.iloc[:, 0])

# get the correlation between features and label (SalePrice)
# Only last column or last row of the correlation matrix above is needed as we want to focus on the label
df_corr = df_corr.iloc[:, 0]
golden_features_list = df_corr[abs(df_corr) > 0.1].sort_values(ascending=False)
print("There is {} strongly correlated values with Diabetes:\n{}".format(len(golden_features_list), golden_features_list))

print(golden_features_list)
print(len(golden_features_list))


import statsmodels.api as sm
import matplotlib.pyplot as plt

df2 = df[df['Diabetes_012'] != 1]
df2['Diabetes_012'] = df['Diabetes_012'].replace(2, 1)

fig, axes = plt.subplots(round((len(golden_features_list) + 2) / 3), 3, figsize = (18, 18))

import statsmodels.api as sm

for i, feature in enumerate(golden_features_list.index):  # Iterate over golden_features_list.index
    try:  # Handle cases where there are fewer golden features than axes
        ax = axes.flatten()[i] # Get the subplot corresponding to the feature
    except IndexError:
        break  # If we run out of subplots, end the loop

    X = df2[[feature]]  # Use double brackets for selecting a column for statsmodels
    y = df2['Diabetes_012']
    X = sm.add_constant(X)
    logit_model = sm.Logit(y, X).fit(disp=0)

    ax.set_xlabel(feature)  # Updated set_xlabel
    ax.set_ylabel('Diabetes')
    ax.set_title(f'Logistic Regression: {feature} vs Diabetes')  # Updated set_title

from scipy.stats import skew

numeric_feats = df.dtypes[df.dtypes != "object"].index

# Check the skew of all numerical features
skewed_feats = df[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
print("\nSkew in numerical features: \n")
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness.head(10)

skewness = skewness[abs(skewness) > 0.75]
print("There are {} skewed numerical features to Box Cox transform".format(skewness.shape[0]))

from scipy.special import boxcox1p
skewed_features = skewness.index
lam = 0.15
for feat in skewed_features:
    df[feat] += 1
    df[feat] = boxcox1p(df[feat], lam)

## FEATURE SELECTION

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import mean_absolute_error
import numpy as np

df = pd.read_csv('diabetes_012_health_indicators.csv')

X = df.drop('Diabetes_012', axis=1)
y = df.Diabetes_012

print(X)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

def average_error_rate(test, pred):
    return (test != pred).mean()

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
model.fit(X_train, y_train)

# predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

print("Training mean absolute error is: ", mean_absolute_error(y_train, y_pred_train))
print("Test mean absolute error is: ", mean_absolute_error(y_test, y_pred_test))
print()
print("Training mean error rate is: ", average_error_rate(y_train, y_pred_train))
print("Test mean error rate is: ", average_error_rate(y_test, y_pred_test))

#X_train_w_label = X_train.copy()
df['Diabetes_012'] = y_train

correlations = df.corr()['Diabetes_012']

# Filter features with correlation above the threshold
golden_features_list = correlations[abs(correlations) >= 0.1].sort_values(ascending=False)

# Remove 'SalePrice' itself from the list
golden_features_list = golden_features_list.drop('Diabetes_012')

print(golden_features_list)

# Create a pipeline with the imputer and the Linear Regression model
model_selected_features = LogisticRegression(multi_class='multinomial', solver='lbfgs')

selected_feature_indices = [X.columns.get_loc(col) for col in golden_features_list.index]

# Fit the pipeline on the training data with selected features
model_selected_features.fit(X_train[:, selected_feature_indices], y_train)

# Make predictions
y_pred_train_ = model_selected_features.predict(X_train[:, selected_feature_indices])
y_pred_test_ = model_selected_features.predict(X_test[:, selected_feature_indices])
print("Training mean absolute error is: ", mean_absolute_error(y_train, y_pred_train_))
print("Test mean absolute error is: ", mean_absolute_error(y_test, y_pred_test_))
print()
print("Training mean error rate is: ", average_error_rate(y_train, y_pred_train_))
print("Test mean error rate is: ", average_error_rate(y_test, y_pred_test_))

print(golden_features_list)

## ! pip install mlxtend (if needed)

# Import sequenctial features selector as SFS
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.plotting import plot_sequential_feature_selection as plot_sfs

# use SequentialFeatureSelector to run a step forward feature selection
# you can set the parameters: features - up to 70, scoring -'neg_mean_absolute_error', cv=3
# Note: you can also use a larger n_jobs to speed up computation
# Create a Linear Regression model
# Create a pipeline with the imputer and the Linear Regression model
regr = LogisticRegression(multi_class='multinomial', solver='lbfgs')
estimator = Pipeline([('imputer', SimpleImputer(strategy='mean')), ('regressor', regr)])

# Create the SequentialFeatureSelector object
sfs = SFS(estimator,  # Use the pipeline as the estimator
          k_features=10,
          forward=True,
          floating=False,
          scoring='neg_mean_absolute_error',
          cv=3,
          n_jobs=-1)

# Fit the SFS object to the training data
sfs = sfs.fit(X_train, y_train)

fig = plot_sfs(sfs.get_metric_dict(), kind='std_err', figsize=(15, 12))

plt.title('Sequential Forward Selection (w. StdErr)')
plt.grid()
plt.show()

# use SequentialFeatureSelector to run a step backward feature selection
# you can set the parameters: features - down to 1, scoring -'neg_mean_absolute_error', cv=3
# Note: you can also use a larger n_jobs to speed up computation
regr = LogisticRegression(multi_class='multinomial', solver='lbfgs')
estimator = Pipeline([('imputer', SimpleImputer(strategy='mean')), ('regressor', regr)])

sfs_backward = sfs_backward = SFS(estimator,
                   k_features=2,
                   forward=False,
                   floating=False,
                   scoring='neg_mean_absolute_error',
                   cv=3,
                   n_jobs=-1)

sfs_backward = sfs_backward.fit(X_train, y_train)

fig = plot_sfs(sfs_backward.get_metric_dict(), kind='std_err', figsize=(15, 12))

plt.title('Sequential Reverse Selection (w. StdErr)')
plt.grid()
plt.show()

# L1-Based Feature Selection
from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel

X = df.drop(columns='Diabetes_012')
y = df['Diabetes_012']

lsvc = LinearSVC(C=0.01, penalty="l1", dual=False).fit(X, y)
model = SelectFromModel(lsvc, prefit=True)
X_new = model.transform(X)

X_new

X_new.shape

X_train, X_test, y_train, y_test = train_test_split(X_new, y, test_size=0.25, random_state=42)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

def average_error_rate(test, pred):
    return (test != pred).mean()

model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
model.fit(X_train, y_train)

# predictions
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

print("Training mean absolute error is: ", mean_absolute_error(y_train, y_pred_train))
print("Test mean absolute error is: ", mean_absolute_error(y_test, y_pred_test))
print()
print("Training mean error rate is: ", average_error_rate(y_train, y_pred_train))
print("Test mean error rate is: ", average_error_rate(y_test, y_pred_test))

## Tree-Based Feature Selection
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel

X = df.drop(columns='Diabetes_012')
y = df['Diabetes_012']

clf = ExtraTreesClassifier(n_estimators=50)
clf = clf.fit(X, y)
print(clf.feature_importances_)

model = SelectFromModel(clf, prefit=True)
X_tree = model.transform(X)

X_tree

sorted(clf.feature_importances_, reverse=True)

## According to Tree Selection, features with descending importance:

## - BMI (3)
## - Age (18)
## - Income (20)
## - PhysHlth (15)
## - MenHlth (14)
## - Education (19)
## - GenHlth (13)
## - HighBP (0)
## - Fruits (8)
## ...
