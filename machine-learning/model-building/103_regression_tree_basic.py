########################################
# %% # Regression Tree - Basic Template
########################################

import pandas as pd
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold
import matplotlib.pyplot as plt

my_df = pd.read_csv('data/sample_data_regression.csv')

X = my_df.drop(["output"], axis=1)
y = my_df.output

# Regression Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

regressor = DecisionTreeRegressor()
regressor.fit(X_train, y_train)

# Assess model accuracy
y_pred = regressor.predict(X_test)
r2_score(y_test,y_pred)

## Cross Validation
cv_scores = cross_val_score(regressor, X, y, cv=4, scoring="r2")

cv = KFold(n_splits=4, shuffle=True, random_state=42)
cv_scores = cross_val_score(regressor, X, y, cv=cv, scoring="r2")
cv_scores.mean()

plt.figure(figsize=(25,15))
tree = plot_tree(regressor,
                 feature_names = X.columns,
                 filled = True,
                 rounded = True,
                 fontsize = 24)
