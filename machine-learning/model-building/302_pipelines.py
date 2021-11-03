########################################
# Pipelines - Basic Template
########################################

# %% # imports
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# import sample data
my_df = pd.read_csv('data/pipeline_data.csv')

# split data into input and output
X = my_df.drop(["purchase"], axis=1)
y = my_df.purchase

# split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# specify numeric and categorical features

numeric_features = list(X.select_dtypes('number').columns)
categorical_features = list(X.select_dtypes('object').columns)

########################################
# Set Up Pipelines
########################################

# Numerical Feature Transformer

numeric_transformer = Pipeline(steps = [("imputer", SimpleImputer()),
                                        ("scaler", StandardScaler())])
# Categorical Feature Transformer
categorical_transformer = Pipeline(steps = [("imputer", SimpleImputer(strategy="constant", fill_value='U')),
                                            ("ohe", OneHotEncoder(handle_unknown="ignore"))])

# Preprocessing Pipeline
preprocessing_pipeline = ColumnTransformer([("numeric", numeric_transformer, numeric_features),
                                            ("categorical", categorical_transformer, categorical_features)])

########################################
# Apply the Pipeline
########################################

# Logistic Regression
clf = Pipeline(steps = [("preprocessing_pipeline", preprocessing_pipeline),
                        ("classifier", LogisticRegression(random_state=42))])

clf.fit(X_train, y_train)
y_pred_class = clf.predict(X_test)
print(accuracy_score(y_test, y_pred_class))
# %%
# Random Forest 
clf = Pipeline(steps = [("preprocessing_pipeline", preprocessing_pipeline),
                        ("classifier", RandomForestClassifier(random_state=42))])
 
clf.fit(X_train, y_train)
y_pred_class = clf.predict(X_test)
print(accuracy_score(y_test, y_pred_class))

# %%
import joblib
joblib.dump(clf, "data/model.joblib")

########################################
# %% # Apply the Pipeline
########################################

import joblib
import pandas as pd
import numpy as np

clf = joblib.load("data/model.joblib")

# create new data
new_data = pd.DataFrame({"age": [25, np.nan, 50],
                         "gender": ["M", "F", np.nan],
                         "credit_score": [200., 100, 500]})

# pass new data in and recieve predictions
clf.predict(new_data)
