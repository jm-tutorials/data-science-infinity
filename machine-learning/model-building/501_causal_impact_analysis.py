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
        t.transaction_date,
        cc.signup_flag,
        sum(t.sales_cost) as sales_cost
    from groceries.transaction_details t
    join groceries.customer_transactions ct on t.transaction_id = ct.transaction_id
    join groceries.customer_campaign cc on ct.customer_id = cc.customer_id
    group by ct.customer_id,
        t.transaction_date,
        cc.signup_flag
)
select
    transaction_date,
    sum(case when signup_flag then sales_cost else 0 end) / sum(case when signup_flag then 1 else 0 end) as member,
    sum(case when not signup_flag then sales_cost else 0 end) / sum(case when not signup_flag then 1 else 0 end) as non_member
from customer_sales
group by
    transaction_date
order by transaction_date
"""
casual_impact_df = pd.read_sql_query(query, con=engine, index_col='transaction_date')

# format for CausalImpact class
casual_impact_df.index = pd.to_datetime(casual_impact_df.index)
casual_impact_df.index.freq = 'D'

########################################
# %% # Apply Causal Impact
########################################

pre_period =[pd.to_datetime("2020-04-01"), pd.to_datetime("2020-06-30")]
post_period =[pd.to_datetime("2020-07-01"), pd.to_datetime("2020-09-30")]

ci = CausalImpact(casual_impact_df, pre_period, post_period)

# plot the impact
ci.plot()

# extract the summary statistics and report
# %%
print(ci.summary())
print()
print(ci.summary(output = "report"))
