# %% # Model Validation

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
# %%
my_df = pd.read_csv('data/sample_data_regression.csv')

X = my_df.drop(["output"], axis=1)
y = my_df.output

# Instantiate our gridsearch object

gscv = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid={"n_estimators": [10, 50, 100, 500],
    "max_depth": [1,23,4,5,6,7,8,9,10]},
    cv=5,
    scoring="r2",
    n_jobs=-1
)

# fit data
gscv.fit(X,y)

# Get the best CV score (mean)
gscv.best_score_

# Optimal Parameters
regressor = gscv.best_estimator_
# %%
