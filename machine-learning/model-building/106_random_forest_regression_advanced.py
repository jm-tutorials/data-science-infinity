########################################
# Regression Tree - ABC Grocery Task
########################################

# %% # Imports
import os
import pickle

import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

from sklearn.ensemble import RandomForestRegressor
from sklearn.utils import shuffle
from sklearn.metrics import r2_score
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

query = "select * from loyalty_score_regression_data"
df = shuffle(pd.read_sql_query(query, con=engine), random_state=42)

# split dataset
data_for_model = df[df.loyalty_score.notna()]
data_for_scoring = df[df.loyalty_score.isna()]

########################################
# %% # check for missing values
########################################
# data_for_model.isna().sum()
data_for_model.dropna(how="any", inplace=True)

X = data_for_model.drop(["customer_id","loyalty_score"], axis=1)
y = data_for_model.loyalty_score

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
regressor = RandomForestRegressor(random_state=42)
regressor.fit(X_train, y_train)

########################################
# %% Model Assesment
########################################

y_pred = regressor.predict(X_test)
r_squared = r2_score(y_test,y_pred)
print('r-squared:',r_squared)

## Cross Validation
cv = KFold(n_splits=4, shuffle=True, random_state=42)
cv_scores = cross_val_score(regressor, X_train, y_train, cv=cv, scoring="r2")
print(cv_scores.mean())

########################################
# Calculate Adjusted R-Squared
########################################

num_data_points, num_input_vars = X_test.shape
adjusted_r_squared = 1 - (1 - r_squared) * (num_data_points - 1) / (num_data_points - num_input_vars)
print(adjusted_r_squared)

# %% # Feature Importance
feature_importance_summary = pd.DataFrame([{"input_variable": col, "feature_importance": importance} 
            for col, importance in zip(X.columns,regressor.feature_importances_)]) \
                .sort_values("feature_importance")

plt.barh(feature_importance_summary.input_variable,feature_importance_summary.feature_importance)
plt.title("Feature Importance of Random Forest Model")
plt.xlabel("Feature Importance")
plt.tight_layout()
plt.show()
# %% # Permutation Importance

result = permutation_importance(regressor, X_test, y_test, n_repeats=10, random_state=42)

permutation_importance_summary = pd.DataFrame([{"input_variable": col, "permutation_importance": importance} 
            for col, importance in zip(X.columns,result['importances_mean'])]) \
                .sort_values("permutation_importance")

plt.barh(permutation_importance_summary.input_variable,permutation_importance_summary.permutation_importance)
plt.title("Permutation Importance of Random Forest Model")
plt.xlabel("Permutation Importance")
plt.tight_layout()
plt.show()

# Write models to pickle files
pickle.dump(regressor, open("data/random_forest_regression_model.p", "wb"))
pickle.dump(one_hot_encoder, open("data/random_forest_regression_ohe.p", "wb"))

# %% Predict Missing Loyalty Scores
data_for_scoring.dropna(how="any", inplace=True)
data_for_scoring = generate_dummy_vars_df(data_for_scoring.drop(["customer_id"], axis=1))
loyalty_predictions = regressor.predict(data_for_scoring)
data_for_scoring['loyalty_predictions'] = loyalty_predictions
