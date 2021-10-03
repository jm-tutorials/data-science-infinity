########################################
# Random Forest for Classification - ABC Grocery Task
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
from sklearn.preprocessing import OneHotEncoder
from sklearn.inspection import permutation_importance

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
# %% # Check class balance
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

clf = RandomForestClassifier(random_state=42, n_estimators=500, max_features=5)
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

########################################
# %% # Feature Selection
########################################

# Feature Importance
feature_importance_summary = pd.DataFrame([{"input_variable": col, "feature_importance": importance} 
            for col, importance in zip(X.columns,clf.feature_importances_)]) \
                .sort_values("feature_importance")

plt.barh(feature_importance_summary.input_variable,feature_importance_summary.feature_importance)
plt.title("Feature Importance of Random Forest Model")
plt.xlabel("Feature Importance")
plt.tight_layout()
plt.show()

# Permutation Importance

result = permutation_importance(clf, X_test, y_test, n_repeats=10, random_state=42)

permutation_importance_summary = pd.DataFrame([{"input_variable": col, "permutation_importance": importance} 
            for col, importance in zip(X.columns,result['importances_mean'])]) \
                .sort_values("permutation_importance")

plt.barh(permutation_importance_summary.input_variable,permutation_importance_summary.permutation_importance)
plt.title("Permutation Importance of Random Forest Model")
plt.xlabel("Permutation Importance")
plt.tight_layout()
plt.show()