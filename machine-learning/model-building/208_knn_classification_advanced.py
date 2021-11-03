########################################
# KNN Classifier - ABC Groceries
########################################

# %% # Imports
import os
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors._classification import KNeighborsClassifier as knnType
from sklearn.utils import shuffle
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.feature_selection import RFECV
from sklearn.ensemble import RandomForestClassifier

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

print(f"Class Balance:\n{data_for_model.signup_flag.value_counts(normalize=True)}")

########################################
# %% # check for missing values
########################################
data_for_model.isna().sum()
data_for_model.dropna(how="any", inplace=True)

########################################
# %% # check for outliers
########################################

def boxplot_outliers(df_col, factor=1.5) -> float: 
    lower_quartile = df_col.quantile(0.25)
    upper_quartile = df_col.quantile(0.75)
    iqr = upper_quartile - lower_quartile
    iqr_extended = iqr * factor
    return lower_quartile - iqr_extended, upper_quartile + iqr_extended

def std_outliers(df_col, factor=None) -> float:
    mean = df_col.mean()
    std_dev = df_col.std()
    return mean - (std_dev * 3), mean + (std_dev * 3)

def drop_outliers(df, cols, method, factor=1.5) -> pd.DataFrame:
    methods = {"boxplot": boxplot_outliers, 'std': std_outliers}
    for col in cols:
        min_border, max_border = methods[method](df[col])
        outliers = df[(df[col] < min_border) | (df[col] > max_border)].index
        print(f"{len(outliers)} outliers detected in column {col}")
        return df.drop(outliers)

outlier_columns = ['distance_from_store', 'total_sales', 'total_items']
data_for_model_2 = drop_outliers(data_for_model, outlier_columns, 'boxplot', factor=2)


########################################
# %% # Split data
########################################

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

def generate_dummy_vars_df(X_df) -> pd.DataFrame:
    X_encoded_array = one_hot_encoder.transform(X_df[categorical_vars])
    X_encoded = pd.DataFrame(X_encoded_array, columns=encoder_features_names)
    return pd.concat([X_df.reset_index(drop=True), X_encoded.reset_index(drop=True)], axis=1) \
        .drop(categorical_vars, axis=1)

X_train = generate_dummy_vars_df(X_train)
X_test = generate_dummy_vars_df(X_test)

########################################
# %% # Feature Scaling
########################################
scale_norm = MinMaxScaler()
scale_norm.fit(X_train)

def normalize_df(df) -> pd.DataFrame:
    return pd.DataFrame(scale_norm.transform(df), columns=df.columns)

X_train = normalize_df(X_train)
X_test = normalize_df(X_test)

########################################
# %% # Recursive Feature Selection 
########################################

clf = RandomForestClassifier(random_state=42)
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

clf = KNeighborsClassifier()
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

# Recall: of all observations that were positive, how many were predicted to be positive
print("recall:", recall_score(y_test, y_pred_class))

# f1-score: the harmonic mean of precision and recall
print("f1-score:",f1_score(y_test, y_pred_class))

########################################
# %% Finding the optimal k
########################################

def fit_decision_tree(k) -> tuple:
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)
    accuracy = f1_score(y_test,y_pred)
    print(f"F1 score {accuracy} at k {k}")
    return clf, accuracy

def get_best_k() -> tuple:
    accuracy_dict = {i: fit_decision_tree(i)[1] for i in range(2,25)} 
    optimal_k_value = max(accuracy_dict, key=accuracy_dict.get)
    max_accuracy = accuracy_dict[optimal_k_value]
    plt.plot(accuracy_dict.keys(), accuracy_dict.values())
    plt.scatter(optimal_k_value, max_accuracy, color='red')
    plt.title(f"Accuracy (F1 Score) by k\n Optimal Tree k: {optimal_k_value} (Accuracy: {round(max_accuracy,3)}")
    plt.xlabel("k")
    plt.ylabel("Accuracy (F1 Score)")
    plt.tight_layout()
    plt.show()
    return optimal_k_value, max_accuracy

k, accuracy = get_best_k()