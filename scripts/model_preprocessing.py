from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
import numpy as np
from sklearn.model_selection import StratifiedKFold
import pickle
import yaml

with open('datasets/stage2/X_Train.npy', 'rb') as f:
    X_Train = np.load(f, allow_pickle=True)
with open('datasets/stage2/y_Train.npy', 'rb') as f:
    y_Train = np.load(f, allow_pickle=True)

params = yaml.safe_load(open("params.yaml"))["train"]
penalty= params["C"]
coef0= params["coef0"]
degree= params["degree"]
gamma= params["gamma"]

C=penalty
kernel = 'poly'
coef0=coef0
degree=degree
gamma=gamma

SVM_clf=SVC(kernel=kernel, C=C,
            degree = degree, coef0 = coef0)

SVM_clf.fit(X_Train, y_Train)

with open("models/model.pkl", "wb") as m:
    pickle.dump(SVM_clf, m)