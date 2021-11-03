########################################
# Classification Tree - ABC Grocery Task
########################################

# %% # Imports
import os
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import OneHotEncoder

########################################
# %% # Read in and shuffle Data
########################################

USER_LOGIN = 'postgres' 
USER_PASSWORD = os.environ.get("POSTGRES_PASSWORD")
SERVICE_NAME = 'localhost'
PORT = 5432
DATABASE = 'postgres'

conn_url = f"postgresql+psycopg2://{USER_LOGIN}:{USER_PASSWORD}@{SERVICE_NAME}:{PORT}/{DATABASE}"
engine = create_engine(conn_url)

query = "select * from groceries.signup_classification_data"
data_for_model = shuffle(pd.read_sql_query(query, con=engine), random_state=42)


########################################
# %% # check for missing values
########################################

print(data_for_model.signup_flag.value_counts(normalize=True))

########################################
# %% # check for missing values
########################################
data_for_model.isna().sum()
data_for_model.dropna(how="any", inplace=True)


########################################
# %% # Split Input Variables & Output Variables
########################################

X = data_for_model.drop(["customer_id","signup_flag"], axis=1)
y = data_for_model.signup_flag
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

########################################
# Deal with Categorical Variables
########################################

categorical_vars = ["gender"]
one_hot_encoder = OneHotEncoder(sparse=False, drop="first")
one_hot_encoder.fit(X_train[categorical_vars])

encoder_features_names = one_hot_encoder.get_feature_names(categorical_vars)

def generate_dummy_vars_df(X_df):
    X_encoded_array = one_hot_encoder.transform(X_df[categorical_vars])
    X_encoded = pd.DataFrame(X_encoded_array, columns=encoder_features_names)
    return pd.concat([X_df.reset_index(drop=True), X_encoded.reset_index(drop=True)], axis=1) \
        .drop(categorical_vars, axis=1)

X_train = generate_dummy_vars_df(X_train)
X_test = generate_dummy_vars_df(X_test)

########################################
#  %% # Model Training
########################################

clf = DecisionTreeClassifier(random_state=42, max_depth=5)
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

# f1-score: the harmonic mean of precision and recall
print("f1-score:",f1_score(y_test, y_pred_class))

########################################
# %% Finding the optimal threshold
########################################

thresholds = np.arange(0, 1, 0.01)

def get_clf_scores(threshold):
    pred_class = (y_pred_prob >= threshold) * 1
    precision = precision_score(y_test, pred_class, zero_division=0)
    recall = recall_score(y_test, pred_class, zero_division=0)
    f1= f1_score(y_test, pred_class, zero_division=0)
    return {'threshold': threshold, 'precision': precision, 'recall': recall, 'f1':f1}

clf_scores = pd.DataFrame([get_clf_scores(threshold) for threshold in thresholds])
optimal_threshold = clf_scores.loc[clf_scores.f1.idxmax(),'threshold']

plt.plot(clf_scores.threshold, clf_scores.precision, label='Precision', linestyle = "--")
plt.plot(clf_scores.threshold, clf_scores.recall, label='Recall', linestyle = "--")
plt.plot(clf_scores.threshold, clf_scores.f1, label='F1', linewidth=5)
plt.title(f"Finding the Optimal Treshold for Classication Model\n Max F1: {round(clf_scores.f1.max(),2)} (Threshold: {optimal_threshold}")
plt.legend(loc="lower left")
plt.tight_layout()
plt.show()

# finding the best max_depth 

y_pred_class = (y_pred_prob >= optimal_threshold) * 1

def fit_decision_tree(depth):
    regressor = DecisionTreeClassifier(max_depth=depth, random_state=42)
    regressor.fit(X_train,y_train)
    y_pred = regressor.predict(X_test)
    accuracy = f1_score(y_test,y_pred)
    print(f"F1 score {accuracy} at depth {depth}")
    return regressor, accuracy

def get_max_depth_list():
    accuracy_dict = {i: fit_decision_tree(i)[1] for i in range(1,9)} 
    optimal_depth = max(accuracy_dict, key=accuracy_dict.get)
    max_accuracy = accuracy_dict[optimal_depth]
    plt.plot(accuracy_dict.keys(), accuracy_dict.values())
    plt.scatter(optimal_depth, max_accuracy, color='red')
    plt.title(f"Accuracy (F1 Score) by Max Depth\n Optimal Tree Depth: {optimal_depth} (Accuracy: {round(max_accuracy,3)}")
    plt.xlabel("Max Depth of Decision Tree")
    plt.ylabel("Accuracy (F1 Score)")
    plt.tight_layout()
    plt.show()
    return optimal_depth, max_accuracy

depth, accuracy = get_max_depth_list()

# retrain model at optimal depth
clf, r_squared = fit_decision_tree(depth)
plt.figure(figsize=(25,15))
tree = plot_tree(clf,
                 feature_names = X.columns,
                 filled = True,
                 rounded = True,
                 fontsize = 24)
tree
