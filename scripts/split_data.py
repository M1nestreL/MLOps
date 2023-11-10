from sklearn.preprocessing import OneHotEncoder, MinMaxScaler # Импортируем нормализацию и One-Hot Encoding от scki-kit-learn
from sklearn.pipeline import Pipeline # Pipeline. Ни добавить, ни убавить
from sklearn.compose import ColumnTransformer # т.н. преобразователь колонок
from sklearn.model_selection import train_test_split
import os
import yaml
import pandas as pd
import numpy as np

X = pd.read_csv('datasets/stage1/X.csv')
y = pd.read_csv('datasets/stage1/y.csv')

params = yaml.safe_load(open("params.yaml"))["split"]
p_split_ratio = params["split_ratio"]

cat_columns = []
num_columns = []
for column_name in X.columns:
    if (X[column_name].dtypes == object):
        cat_columns +=[column_name]
    else:
        num_columns +=[column_name]
# Pipeline для числовых данных (нормализация)
numerical_pipe = Pipeline([
    ('scaler', MinMaxScaler())
])
# Pipeline для категориальных данных (One-Hot кодирование)
categorical_pipe = Pipeline([
    ('encoder', OneHotEncoder(drop='if_binary', handle_unknown='ignore', sparse_output=False))
])

preprocessors = ColumnTransformer(transformers=[
    ('num', numerical_pipe, num_columns),
    ('cat', categorical_pipe, cat_columns)
])
preprocessors.fit(X)
X_train = preprocessors.transform(X) # Преобразуем  тренировочные данные

# Разбиваем тренировочные данные на тренировочную и валидационную выборку
X_Train, X_val, y_Train, y_val = train_test_split(X_train, y.values.ravel(), test_size=p_split_ratio, random_state=44)
print(X_Train.shape, X_val.shape, y_Train.shape, y_val.shape)

os.makedirs(os.path.join("datasets", "stage2"),exist_ok=True)

with open('datasets/stage2/X_Train.npy', 'wb') as f:
    np.save(f, X_Train)
with open('datasets/stage2/X_val.npy', 'wb') as f:
    np.save(f, X_val)
with open('datasets/stage2/y_Train.npy', 'wb') as f:
    np.save(f, y_Train)
with open('datasets/stage2/y_val.npy', 'wb') as f:
    np.save(f, y_val)