import pandas as pd
import os

df = pd.read_csv('datasets/Car_Insurance_Claim.csv')
df.fillna(0, inplace = True)

X, y = df.drop(columns = ['OUTCOME']),df['OUTCOME']

os.makedirs(os.path.join("data", "stage1"),exist_ok=True)

X.to_csv('data/stage1/X.csv', index=False)
y.to_csv('data/stage1/y.csv', index=False)