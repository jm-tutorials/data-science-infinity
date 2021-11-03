# %% # Imports
import os

import pandas as pd
import matplotlib.pyplot as plt
from sqlalchemy import create_engine

from sklearn.linear_model import LinearRegression
from sklearn.utils import shuffle
from sklearn.metrics import r2_score
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
# %% # Recursive Feature Elimination with Cross 
########################################
regressor = LinearRegression()
feature_selector = RFECV(regressor)

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

regressor.fit(X_train, y_train)

########################################
# %% Model Assesment
########################################

y_pred = regressor.predict(X_test)
r_squared = r2_score(y_test,y_pred)

## Cross Validation
cv = KFold(n_splits=4, shuffle=True, random_state=42)
cv_scores = cross_val_score(regressor, X_train, y_train, cv=cv, scoring="r2")
cv_scores.mean()

########################################
# Calculate Adjusted R-Squared
########################################

num_data_points, num_input_vars = X_test.shape
adjusted_r_squared = 1 - (1 - r_squared) * (num_data_points - 1) / (num_data_points - num_input_vars)
print(adjusted_r_squared)

########################################
# %% # Extract Model Coefficients and Intercept
########################################

summary_stats = pd.DataFrame([{'input_variable': col, 'coefficient': coef} 
    for col, coef in zip(X_train.columns, regressor.coef_)])

regressor.intercept_




