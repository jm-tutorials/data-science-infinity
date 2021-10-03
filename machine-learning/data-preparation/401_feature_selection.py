# %%
import pandas as pd
import os
from sklearn.feature_selection import SelectKBest, f_regression, chi2, RFECV
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# %%
my_df = pd.read_csv("feature_selection_sample_data.csv")
X = my_df.drop(["output"], axis=1)
y = my_df["output"]

correlation_matrix = my_df.corr()
# %% ## Regression Template

feature_selector = SelectKBest(f_regression, k='all')
fit = feature_selector.fit(X,y)

summary_stats = pd.DataFrame([
    {"input_variable": column_name , "p_value": p, "f_score": f}
    for column_name, p, f in zip(X.columns, fit.pvalues_,fit.scores_)
    ]).sort_values(by="p_value")

p_values_threshold = 0.05
f_score_threshold = 5

selected_variables = summary_stats.loc[(summary_stats.f_score >= f_score_threshold) \
    & (summary_stats.p_value <= p_values_threshold)]
selected_variables = selected_variables.input_variable.tolist()
X_new = X[selected_variables]

# %% ## Classification Template

feature_selector = SelectKBest(chi2, k='all')
fit = feature_selector.fit(X,y)

summary_stats = pd.DataFrame([
    {"input_variable": column_name , "p_value": p, "ch2_score": chi}
    for column_name, p, chi in zip(X.columns, fit.pvalues_,fit.scores_)
    ]).sort_values(by="p_value")

p_values_threshold = 0.05
f_score_threshold = 5

selected_variables = summary_stats.loc[(summary_stats.f_score >= f_score_threshold) \
    & (summary_stats.p_value <= p_values_threshold)]
selected_variables = selected_variables.input_variable.tolist()
X_new = X[selected_variables]

# %% # Recursive Feature Elimination with Cross 

regressor = LinearRegression()
feature_selector = RFECV(regressor)

fit = feature_selector.fit(X,y)
optimal_feature_count = feature_selector.n_features_
print(f"Optimal number of features: {optimal_feature_count}")

X_new = X.loc[:, feature_selector.get_support()]

plt.plot(range(1, len(fit.grid_scores_) + 1), fit.grid_scores_, marker = "o")
plt.ylabel("Model Score")
plt.xlabel("Number of Features")
plt.title(f"Feature Selection using RFE \n Optimal number of features is {optimal_feature_count} (at score of {round(max(fit.grid_scores_),4)})")
plt.tight_layout()
plt.show()