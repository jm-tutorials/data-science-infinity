# %%
import pandas as pd

my_df = pd.DataFrame({"input1": [15,41,44,47,50,53,56,59,99],
                      "input2": [29,41,44,47,50,53,56,59,66]})
outlier_columns=["input1", "input2"]
# %%
my_df.plot(kind="box", vert=False)
# %%
def boxplot_outliers(df_col): 
    lower_quartile = df_col.quantile(0.25)
    upper_quartile = df_col.quantile(0.75)
    iqr = upper_quartile - lower_quartile
    iqr_extended = iqr * 1.5
    return lower_quartile - iqr_extended, upper_quartile + iqr_extended

def std_outliers(df_col):
    mean = df_col.mean()
    std_dev = df_col.std()
    return mean - (std_dev * 3), mean + (std_dev * 3)

def drop_outliers(df, cols, method):
    methods = {"boxplot": boxplot_outliers, 'std': std_outliers}
    for col in cols:
        min_border, max_border = methods[method](df[col])
        outliers = df[(df[col] < min_border) | (df[col] > max_border)].index
        print(f"{len(outliers)} outliers detected in column {col}")
        return df.drop(outliers)

if __name__ == "__main__":
    cleaned_df1 = drop_outliers(my_df, outlier_columns, "boxplot")
    cleaned_df2 = drop_outliers(my_df, outlier_columns, "std")
    
