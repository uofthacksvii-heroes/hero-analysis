import pandas as pd
import pickle
from sklearn.linear_model import LogisticRegression

df = pd.read_csv('data_one.csv')
features = df.columns[:2]
x = df.iloc[:, :2]
y = df.iloc[:, -1]


# Creating and training the Logistic Regression model
regression = LogisticRegression()
regression.fit(x, y)

# Saving model to disk
pickle.dump(regression, open('model_logistic_regression.pkl', 'wb'))
