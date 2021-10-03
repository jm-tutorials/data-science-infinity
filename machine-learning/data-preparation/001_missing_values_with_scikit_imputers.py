# %%
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer, KNNImputer

# %%
my_df = pd.DataFrame({"A": [1,4,7,10,13],
                      "B": [3,6,9,np.nan,15],
                      "C": [2,5,np.nan,11,np.nan]})

###### SimpleImputer - uses mean of column be default, also has options of mode or a constant value ypu can specify
imputer = SimpleImputer()

imputer.fit(my_df)
imputed_array = imputer.transform(my_df)

# apply fit and trasnform to the same  
# imputer.fit_transform(my_df)

my_df2 = pd.DataFrame(imputer.fit_transform(my_df), columns=my_df.columns)

# %% ###### KNNImputer
my_df = pd.DataFrame({"A": [1,2,3,4,5],
                      "B": [1,1,3,3,4],
                      "C": [1,2,9,np.nan,20]})

knn_imputer = KNNImputer()
knn_imputer = KNNImputer(n_neighbors=1)
knn_imputer = KNNImputer(n_neighbors=2)
knn_imputer = KNNImputer(n_neighbors=2, weights='distance')
knn_imputer.fit_transform(my_df)

my_df2 = pd.DataFrame(knn_imputer.fit_transform(my_df), columns=my_df.columns)