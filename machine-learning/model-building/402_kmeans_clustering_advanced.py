########################################
# K Means Clustering - Advanced Template
########################################

# %%
import os

import pandas as pd
from sqlalchemy import create_engine
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# %% # read in data

USER_LOGIN = "postgres"
USER_PASSWORD = os.environ.get("POSTGRES_PASSWORD")
SERVICE_NAME = "localhost"
PORT = 5432
DATABASE = 'postgres'

conn_url = f"postgresql+psycopg2://{USER_LOGIN}:{USER_PASSWORD}@{SERVICE_NAME}:{PORT}/{DATABASE}"
engine = create_engine(conn_url)

query = """select
    ct.customer_id,
    p.product_area_name,
    sum(t.sales_cost) as sales_cost
from groceries.transaction_details t
join groceries.customer_transactions ct on t.transaction_id = ct.transaction_id
join groceries.product_areas p on t.product_area_id = p.product_area_id
where p.product_area_name != 'Non-Food'
group by ct.customer_id, p.product_area_name
"""
df = pd.read_sql_query(query, con=engine)

########################################
# %% # Data Prep/Cleaning
########################################

# pivot data
transaction_summary_pivot = df.pivot_table(
    index="customer_id",
    columns="product_area_name",
    values="sales_cost",
    aggfunc="sum",
    fill_value=0,
    margins=True,
    margins_name="total") \
        .rename_axis(None, axis=1)

# turn salesx into % of sales
data_for_clustering = transaction_summary_pivot.div(transaction_summary_pivot.total, axis=0) \
    .drop("total", axis=1)

# %%

data_for_clustering.isna().sum()

# normalise data
scale_norm = MinMaxScaler()
data_for_clustering_scaled = pd.DataFrame(
    scale_norm.fit_transform(data_for_clustering),
    columns=data_for_clustering.columns)

########################################
# %% # Use WCCS to find a good value for k
########################################

def fit_kmeans(df, k):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df)
    return kmeans

inertia_by_k = pd.DataFrame([
    {"k": k, "inertia": fit_kmeans(data_for_clustering_scaled, k).inertia_} 
    for k in range(1,10)])

plt.plot("k", "inertia", data=inertia_by_k)
plt.title("Within Cluster Sum of Squares - by k")
plt.xlabel("k")
plt.ylabel("WCSS Score")
plt.tight_layout()
plt.show()

# decided 3 was best value for k based on chart

########################################
# %% # Instantiate and fit model
########################################

kmeans = fit_kmeans(data_for_clustering_scaled, 3)

# %%

data_for_clustering["cluster"] = kmeans.labels_
data_for_clustering["cluster"].value_counts()


########################################
# %% # Profile our clusters
########################################

cluster_summary = data_for_clustering.groupby("cluster")[["Dairy","Fruit","Meat","Vegetables"]].mean()
cluster_summary