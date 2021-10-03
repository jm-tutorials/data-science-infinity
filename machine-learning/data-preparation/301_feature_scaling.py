# %%
import pandas as pd 
from sklearn.preprocessing import StandardScaler, MinMaxScaler

my_df = pd.DataFrame({
    "Height": [1.98,1.77,1.76,1.8,1.64],
    "Weight": [99,81,70,86,82]
})

# %% ## Standardization
scale_standard = StandardScaler()
scale_standard.fit_transform(my_df)
my_df_standardized = pd.DataFrame(scale_standard.fit_transform(my_df), columns=my_df.columns)

# %% ## Normalization
scale_norm = MinMaxScaler()
scale_norm.fit_transform(my_df)
my_df_normalized = pd.DataFrame(scale_norm.fit_transform(my_df), columns=my_df.columns)