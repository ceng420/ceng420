# Breast Cancer Wisconsin dataset

# pandas to load the dataset
from functools import _lru_cache_wrapper

import pandas as pd

# we use this module to encode the labels from B and M to 0 and 1
from sklearn.preprocessing import LabelEncoder

# we use this module to split the data into training and testing
from sklearn.model_selection import train_test_split

# this the ML module we are going to use (evaluate)
from sklearn.linear_model import LogisticRegression

# we use this module to scale or normalize the feature valued between 0 and 1
from sklearn.preprocessing import StandardScaler

# we use this module to do feature reduction
from sklearn.decomposition import PCA

# we build a pipeline to apply preprocessing, encoding, feature extraction, selection, and model evaluation
from sklearn.pipeline import Pipeline

# we use this module to perfrom cross validation
from sklearn.model_selection import cross_val_score

# we use this module to debug the ML model using learning curve
from sklearn.model_selection import learning_curve

# we use numpy for array manipulation
import numpy as np
# we use this module for plotting the learning curve
import matplotlib.pyplot as plt


# the data set has 568 samples (case) some of them have malignant tumors M and some have benign tumors
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data',
                 header=None)

# print the number of samples in the dataset
print(len(df))

# print the number of features in the dataset
print(len(df.columns.values))


# print the name of features in the dataset
print(df.columns.values)

# print the number of features in the dataset
print(df.head(5))


# we select the samples data without the labels (class)
X = df.loc[:, 2:].values

# we select the labels column and encode it into binary values where the malignant tumors are now represented as class 1
# and the benign tumors are represented as class 0

y = df.loc[:, 1].values
le = LabelEncoder()
y = le.fit_transform(y)

# divide the dataset into a separate training dataset (80 percent of the data) and
# a separate test dataset (20 percent of the data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)


# learning algorithms require input features on the same scale for optimal performance
# pipeline is an abstract notion, it is not some existing ml algorithm.
# Often in ML tasks you need to perform sequence of different transformations
# (find set of features, generate new features, select only some good features)
# of raw dataset before applying final estimator.
bcw_pipeline = Pipeline([('scl', StandardScaler()), ('pca', PCA(n_components=2)),
                         ('clf', LogisticRegression(random_state=1))])

bcw_pipeline.fit(X_train, y_train)

print('Test Accuracy: %.3f' % bcw_pipeline.score(X_test, y_test))


scores = cross_val_score(estimator=bcw_pipeline, X=X_train, y=y_train, cv=10, n_jobs=1)

print('CV accuracy scores: %s' % scores)
print('CV accuracy: %.3f +/- %.3f' % (np.mean(scores), np.std(scores)))


# we use learning curve for plotting the training and test accuracies as functions of the sample size
lr_curve_pipeline = Pipeline([('scl', StandardScaler()),
                              ('clf', LogisticRegression(penalty='l2', random_state=0))
                              ])
train_sizes, train_scores, test_scores = learning_curve(estimator=lr_curve_pipeline, X=X_train, y=y_train,
                                                        train_sizes=np.linspace(0.1, 1.0, 10), cv=10, n_jobs=1)

train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(train_sizes, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
plt.fill_between(train_sizes, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')

plt.plot(train_sizes, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='validation accuracy')
plt.fill_between(train_sizes, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')

plt.grid()
plt.xlabel('Number of training samples')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.ylim([0.8, 1.0])
plt.show()


from sklearn.model_selection import validation_curve

# we vary the values of the model parameters,
# for example, the inverse regularization parameter C in logistic regression
param_range = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
train_scores, test_scores = validation_curve( estimator=lr_curve_pipeline, X=X_train, y=y_train,
                                              param_name='clf__C', param_range=param_range, cv=10)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)
test_std = np.std(test_scores, axis=1)

plt.plot(param_range, train_mean, color='blue', marker='o', markersize=5, label='training accuracy')
plt.fill_between(param_range, train_mean + train_std, train_mean - train_std, alpha=0.15, color='blue')
plt.plot(param_range, test_mean, color='green', linestyle='--', marker='s', markersize=5, label='validation accuracy')
plt.fill_between(param_range, test_mean + test_std, test_mean - test_std, alpha=0.15, color='green')

plt.grid()
plt.xscale('log')
plt.legend(loc='lower right')
plt.xlabel('Parameter C')
plt.ylabel('Accuracy')
plt.ylim([0.8, 1.0])
plt.show()