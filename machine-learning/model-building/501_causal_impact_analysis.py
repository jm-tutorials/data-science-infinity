import os 

from causalimpact import CausalImpact
import pandas as pd
from sqlalchemy import create_engine

from sklearn.preprocessing import OneHotEncoder

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

query = """with t as (
    select 
        customer_id,
        transaction_date,
        sum(sales_cost) as sales_cost
    from groceries.transactions
    group by customer_id,
        transaction_date
)
select

from 

"""
df = pd.read_sql_query(query, con=engine)

