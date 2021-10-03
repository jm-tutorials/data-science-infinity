# %% # Model Validation

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
# %%
my_df = pd.read_csv('feature_selection_sample_data.csv')

X = my_df.drop(["output"], axis=1)
y = my_df.output

# Regression Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
r2_score(y_test,y_pred)

# Classifaiction Model - add stratify option to keep proportions of y variable consistant in both the traning and testing sets 
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

## Cross Validation
# %% # Regression
cv_scores = cross_val_score(regressor, X, y, cv=4, scoring="r2")

cv = KFold(n_splits=4, shuffle=True, random_state=42)
cv_scores = cross_val_score(regressor, X, y, cv=cv, scoring="r2")
cv_scores.mean()

# %% # classification
#cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=42)
#cv_scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
#cv_scores.mean()
