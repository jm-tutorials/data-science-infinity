# %% # Model Validation

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import matplotlib.pyplot as plt

my_df = pd.read_csv('data/sample_data_classification.csv')

X = my_df.drop(["output"], axis=1)
y = my_df.output

# Classification Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

clf = LogisticRegression(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
accuracy_score(y_test,y_pred)

y_pred_prob = clf.predict_proba(X_test)

# Confusion Matrix

conf_matrix = confusion_matrix(y_test, y_pred)

plt.style.use("seaborn-poster")
plt.matshow(conf_matrix, cmap='coolwarm')
plt.gca().xaxis.tick_bottom()
plt.title("Confusion Matrix")
plt.ylabel("Actual Class")
plt.xlabel("Predicted Class")
for (i, j), corr_value in np.ndenumerate(conf_matrix):
    plt.text(j, i, corr_value, ha="center", va="center", fontsize=20)
plt.show()