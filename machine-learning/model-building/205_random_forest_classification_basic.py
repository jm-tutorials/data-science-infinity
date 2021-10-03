########################################
# %% # Random Forest Classification - Basic Template
########################################

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import matplotlib.pyplot as plt

my_df = pd.read_csv('data/sample_data_classification.csv')

X = my_df.drop(["output"], axis=1)
y = my_df.output

# Regression Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

# Assess model accuracy
y_pred = clf.predict(X_test)
print("test:",accuracy_score(y_test,y_pred))

# Assess model accuracy
y_pred_training = clf.predict(X_train)
print("train:",accuracy_score(y_train,y_pred_training))
