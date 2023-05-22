print("Importing libraries...")
import pandas as pd
print("reading data...")

#data_en_train = pd.read_csv("data_ready_en_train.csv")
#data_en_train_oversampled = pd.read_csv("data_ready_en_train_oversampled.csv")
data_en_train_undersampled = pd.read_csv("../datasets/data_ready_en_train_undersampled.csv")
data_en_test = pd.read_csv("../datasets/data_ready_en_test.csv")
#%%
X_test = data_en_test.drop(['passed','p_recall'], axis=1)
y_test = data_en_test['passed']
X_test

import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

# X = data_en_train.drop('passed', axis=1)
# y = data_en_train['passed']
# X
print("splitting data")
X_undersampled = data_en_train_undersampled.drop('passed', axis=1)
y_undersampled = data_en_train_undersampled['passed']
X_undersampled

svc_undersampled = SVC(kernel='poly',verbose=3)
print("Fitting...")
svc_undersampled.fit(X_undersampled, y_undersampled)

y_pred = svc_undersampled.predict(X_test)
confusion_matrix(y_test, y_pred)
#%%
roc_auc_score(y_test, svc_undersampled.predict_proba(X_test)[:, 1])