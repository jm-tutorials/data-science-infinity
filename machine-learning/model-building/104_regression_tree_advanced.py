########################################
# Regression Tree - ABC Grocery Task
########################################

# %% # Imports
import os
from numpy.random.mtrand import random_sample 

import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.utils import shuffle
from sklearn.metrics import r2_score
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

X = data_for_model_2.drop(["loyalty_score"], axis=1)
y = data_for_model_2.loyalty_score

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
regressor = DecisionTreeRegressor(random_state=42)
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

########################################
# %% # check for overfitting
########################################

y_train_pred = regressor.predict(X_train)
r_squared = r2_score(y_train,y_train_pred)
print('r-squared on traininsg set:',r_squared)
# here we can see overfitting in the training set, r-squared of 1

def fit_decision_tree(depth):
    regressor = DecisionTreeRegressor(max_depth=depth, random_state=42)
    regressor.fit(X_train,y_train)
    y_pred = regressor.predict(X_test)
    r_squared = r2_score(y_test,y_pred)
    print(f"r-squared {r_squared} at depth {depth}")
    return regressor, r_squared

def get_max_depth_list():
    accuracy_dict = {i: fit_decision_tree(i)[1] for i in range(1,9)} 
    optimal_depth = max(accuracy_dict, key=accuracy_dict.get)
    max_accuracy = accuracy_dict[optimal_depth]
    plt.plot(accuracy_dict.keys(), accuracy_dict.values())
    plt.scatter(optimal_depth, max_accuracy, color='red')
    plt.title(f"Accuracy by Max Depth\n Optimal Tree Depth: {optimal_depth} (Accuracy: {round(max_accuracy,3)}")
    plt.show()
    return optimal_depth, max_accuracy

depth, accuracy = get_max_depth_list()

# retrain model at optimal depth
regressor, r_squared = fit_decision_tree(depth)
plt.figure(figsize=(25,15))
tree = plot_tree(regressor,
                 feature_names = X.columns,
                 filled = True,
                 rounded = True,
                 fontsize = 24)
tree
