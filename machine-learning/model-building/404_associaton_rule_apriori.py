########################################
# Association Rule Apriori
########################################

# %%
from apyori import apriori
import pandas as pd
from pandas.core.dtypes.missing import isna

# read data into list of lists for apriori method

#alcohol_transactions = pd.read_csv("data/sample_data_apriori.csv")
#transactions_list = [[col for col in row if pd.notna(col)]
#    for row in alcohol_transactions.drop('transaction_id', axis=1).values]

with open("data/sample_data_apriori.csv", "r") as f:
    text = f.read().split("\n")
headers = text[0]
data = text[1:]
transactions_list = [[col for col in row.split(",") if col] for row in data]

########################################
# %% # Apply the Apriori algorithm
########################################

apriori_rules = apriori(transactions_list,
    min_support = 0.003,
    min_confidence = 0.2,
    min_lift = 3,
    min_length = 2,
    max_length = 3)

apriori_rules_df = pd.DataFrame([{
    "product_1": list(rule[2][0][0])[0],
    "product_2": list(rule[2][0][1])[0],
    "support": rule[1],
    "confidence": rule[2][0][2],
    "lift": rule[2][0][3]
} for rule in list(apriori_rules)])

# %%
apriori_rules_df.sort_values(by="lift", ascending=False)

# %%
