import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('data_multiple.csv.csv')
features = df.columns[:6]
x = df.iloc[:, :6]
y = df.iloc[:, -1]


# Creating and training the Random Forest model
random_forest = RandomForestClassifier(n_jobs=2, random_state=0)
random_forest.fit(x, y)

# Saving model to disk
pickle.dump(random_forest, open('model_random_forest.pkl', 'wb'))
