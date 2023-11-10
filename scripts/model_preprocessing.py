from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.utils.fixes import loguniform
import yaml
import numpy as np
from sklearn.model_selection import StratifiedKFold
import pickle

with open('data/stage2/X_train.npy', 'rb') as f:
    X_train = np.load(f, allow_pickle=True)
with open('data/stage2/y_Train.npy', 'rb') as f:
    y_Train = np.load(f, allow_pickle=True)
with open('data/stage2/X_val.npy', 'rb') as f:
    X_val = np.load(f, allow_pickle=True)
with open('data/stage2/y_val.npy', 'rb') as f:
    y_val = np.load(f, allow_pickle=True)

C=100

kernel = 'poly'
coef0=1.5
degree=2
gamma=0.2

if gamma == 0:
  gamma='auto'

SVM_clf=SVC(kernel=kernel, C=C,
            degree = degree, coef0 = coef0)

C_range = loguniform(1e-2, 1e2)
gamma_range = loguniform(1e-2, 1e0)
C_range_poly = loguniform(1e-1, 1e1)
tuned_parameters = [#{'kernel': ['rbf'], 'gamma': gamma_range,
                     #'C': C_range,},
                    {'kernel': ['poly', 'rbf', 'linear'], 'degree': [2,3,4,], 'C': C_range_poly, }]


n_iter_search = 15
SVС_search = RandomizedSearchCV(estimator = SVC(coef0=0.5), verbose = 3,
                          param_distributions=tuned_parameters ,
                          cv=StratifiedKFold(n_splits=5, shuffle = True,random_state=42),n_iter = n_iter_search)

SVС_search.fit(X_train, y_Train)

SVCbest=SVС_search.best_estimator_
SVCbest.fit(X_train, y_Train)

with open("models/model.pkl", "wb") as m:
    pickle.dump(SVCbest, m)