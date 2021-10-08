########################################
# PCA - ABC Grocery Task
########################################

# %% # Imports
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.inspection import permutation_importance
from sklearn.decomposition import PCA

# Read in data

data_for_model = shuffle(pd.read_csv("data/sample_data_pca.csv"), random_state=42)

########################################
# %% # Check class balance
########################################

print(data_for_model.purchased_album.value_counts(normalize=True))

########################################
# %% # check for missing values
########################################

data_for_model.isna().sum().sum()
data_for_model.dropna(how="any", inplace=True)

########################################
# %% # Split Input Variables & Output Variables
########################################

X = data_for_model.drop(["user_id","purchased_album"], axis=1)
y = data_for_model.purchased_album
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

########################################
# %% # Feature Scaling
########################################

scale_standard = StandardScaler()

X_train = scale_standard.fit_transform(X_train)
X_train = scale_standard.transform(y_train)

########################################
# %% # PCA
########################################

pca = PCA(n_components=None, random_state=42)
pca.fit(X_train)

explained_variance = pca.explained_variance_ratio_
explained_variance_cumalative = pca.explained_variance_ratio_.cumsum()


########################################
#  %% # Plot the explained variance across componenets
########################################

num_vars_list = list(range(1,101))
plt.figure(figsize=(15,10))

# create list for number of components
plt.subplot(2,1,1)
plt.bar(num_vars_list, explained_variance)
plt.title("Variance Across Principal Components")
plt.xlabel("Number of Components")
plt.ylabel("% Variance")
plt.tight_layout()

plt.subplot(2,1,2)
plt.plot(num_vars_list, explained_variance_cumalative)
plt.title("Cumulative Variance Across Principal Components")
plt.xlabel("Number of Components")
plt.ylabel("Cumalative % Variance")
plt.tight_layout()
plt.show()

########################################
#  %% # Apply PCA with Selected Number of Componenets
########################################

pca = PCA(n_components=0.75, random_state=42)
pca.fit_transform(X_train)
pca.transform(X_test)

########################################
#  %% # Model Training
########################################

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)

########################################
# %% Model Assesment
########################################

y_pred_class = clf.predict(X_test)
y_pred_prob = clf.predict_proba(X_test)[:,1]

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred_class)

plt.style.use("seaborn-poster")
plt.matshow(conf_matrix, cmap='coolwarm')
plt.gca().xaxis.tick_bottom()
plt.title("Confusion Matrix")
plt.ylabel("Actual Class")
plt.xlabel("Predicted Class")
for (i, j), corr_value in np.ndenumerate(conf_matrix):
    plt.text(j, i, corr_value, ha="center", va="center", fontsize=20)
plt.show()

# Accuracy: the number of correct classifiaction out of all attempted classifications
print("accuracy:",accuracy_score(y_test, y_pred_class))

# Precision: of all observations that were predicted as positive, how many were actually positive
print("precision:", precision_score(y_test, y_pred_class))

# Precision: of all observations that were predicted as positive, how many were actually positive
print("recall:", recall_score(y_test, y_pred_class))

# f1-score: the harmonic mean of precision and recall
print("f1-score:",f1_score(y_test, y_pred_class))