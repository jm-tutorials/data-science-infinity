########################################
# Causal Impact Analysis
########################################

# %%
import os 

import pandas as pd
from causalimpact import CausalImpact
from sqlalchemy import create_engine

########################################
# %% # Read in and shuffle Data
########################################
USER_LOGIN = 'postgres' 
USER_PASSWORD = os.environ.get("POSTGRES_PASSWORD")
SERVICE_NAME = 'localhost'
PORT = 5432
DATABASE = 'postgres'

conn_url = f"postgresql+psycopg2://{USER_LOGIN}:{USER_PASSWORD}@{SERVICE_NAME}:{PORT}/{DATABASE}"
engine = create_engine(conn_url)

query = """with customer_sales as (
    select 
        ct.customer_id,
        cc.signup_flag,
        t.transaction_date,
        sum(t.sales_cost) as sales_cost
    from groceries.transaction_details t
    join groceries.customer_transactions ct on t.transaction_id = cc.transaction_id,
    join groceries.customer_campaign cc on t.customer_id = cc.customer_id
    group by ct.customer_id,
        t.transaction_date,
        cc.signup_flag
)
select
    transaction_date,
    sum(case when not signup_flag then sales_cost else 0) / sum(case when not signup_flag then 1 else 0) as not_signed_up_sales,
    sum(case when signup_flag then sales_cost else 0) / sum(case when signup_flag then 1 else 0) as signed_up_sales
from customer_sales
group by
    transaction_date
"""
df = pd.read_sql_query(query, con=engine)


# %%
