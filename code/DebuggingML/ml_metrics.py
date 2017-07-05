import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold, cross_val_score

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# the data set has 568 samples (case) some of them have malignant tumors M and some have benign tumors
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data',
                 header=None)

# we select the samples data without the labels (class)
X = df.loc[:, 2:].values

# we select the labels column and encode it into binary values where the malignant tumors are now represented as class 1
# and the benign tumors are represented as class 0

Y = df.loc[:, 1].values
le = LabelEncoder()
Y = le.fit_transform(Y)

seed = 7
kfold = KFold(n_splits=10, random_state=seed)
model = LogisticRegression()
scoring = 'accuracy'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("Accuracy: mean {0} and std {1}".format(results.mean()*100, results.std()*100))

scoring = 'neg_log_loss'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
# scikit-learn implements metrics this way so that larger is better (i.e., to maximize score).
# scikit-learn unified scoring API always maximizes the score, so scores which need to be minimized
# are negated in order for the unified scoring API to work correctly.
print("Log Loss: mean {0} and std {1}".format(results.mean()*100, results.std()*100))

scoring = 'roc_auc'
results = cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
print("AUC-ROC: mean {0} and std {1}".format(results.mean()*100, results.std()*100))


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=1)
model = LogisticRegression()
model.fit(X_train, y_train)
predicted = model.predict(X_test)
matrix = confusion_matrix(y_test, predicted)
print("Confusion Matrix {0}".format(matrix))