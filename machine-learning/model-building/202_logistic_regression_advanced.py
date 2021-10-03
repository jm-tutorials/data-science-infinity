# %% # Imports
import os
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import RFECV

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

data_for_model.signup_flag.value_counts(normalize=True)

########################################
# %% # check for missing values
########################################
data_for_model.isna().sum()
data_for_model.dropna(how="any", inplace=True)

########################################
# %% # check for outliers
########################################

def boxplot_outliers(df_col, factor=1.5): 
    lower_quartile = df_col.quantile(0.25)
    upper_quartile = df_col.quantile(0.75)
    iqr = upper_quartile - lower_quartile
    iqr_extended = iqr * factor
    return lower_quartile - iqr_extended, upper_quartile + iqr_extended

def std_outliers(df_col, factor=None):
    mean = df_col.mean()
    std_dev = df_col.std()
    return mean - (std_dev * 3), mean + (std_dev * 3)

def drop_outliers(df, cols, method, factor=1.5):
    methods = {"boxplot": boxplot_outliers, 'std': std_outliers}
    for col in cols:
        min_border, max_border = methods[method](df[col])
        outliers = df[(df[col] < min_border) | (df[col] > max_border)].index
        print(f"{len(outliers)} outliers detected in column {col}")
        return df.drop(outliers)

outlier_columns = ['distance_from_store', 'total_sales', 'total_items']
data_for_model_2 = drop_outliers(data_for_model, outlier_columns, 'boxplot', factor=2)
#data_for_model_3 = drop_outliers(data_for_model, outlier_columns, 'std')

X = data_for_model_2.drop(["customer_id","signup_flag"], axis=1)
y = data_for_model_2.signup_flag
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
# %% # Recursive Feature Elimination with Cross 
########################################
clf = LogisticRegression(random_state=42, max_iter=1000)
feature_selector = RFECV(clf)

fit = feature_selector.fit(X_train, y_train)
optimal_feature_count = feature_selector.n_features_
print(f"Optimal number of features: {optimal_feature_count}")

X_new = X.loc[:, feature_selector.get_support()]

plt.plot(range(1, len(fit.grid_scores_) + 1), fit.grid_scores_, marker = "o")
plt.ylabel("Model Score")
plt.xlabel("Number of Features")
plt.title(f"Feature Selection using RFE \n Optimal number of features is {optimal_feature_count} (at score of {round(max(fit.grid_scores_),4)})")
plt.tight_layout()
plt.show()

########################################
#  %% # Model Training
########################################

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

# %% # Accuracy: the number of correct classifiaction out of all attempted classifications
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

y_pred_class = (y_pred_prob >= optimal_threshold) * 1