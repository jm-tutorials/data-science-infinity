create schema if not exists groceries;
set search_path to groceries, public;

-- create tables
create temp table delivery_club_campaign_temp (
    customer_id NUMERIC,
    campaign_name TEXT,
    campaign_date DATE,
    mailer_type TEXT,
    signup_flag BOOLEAN
);

drop table if exists customers cascade; 
create table customers (
    customer_id NUMERIC PRIMARY KEY,
    distance_from_store NUMERIC,
    gender TEXT,
    credit_score NUMERIC
);

drop table if exists loyalty_scores;
create table loyalty_scores (
    customer_id NUMERIC NOT NULL,
    score_datetime TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    customer_loyalty_score NUMERIC,
    primary key (customer_id, score_datetime),
    constraint fk_loyalty_customer_id
        foreign key (customer_id)
            references customers (customer_id)
);

drop table if exists product_areas cascade;
create table product_areas (
    product_area_id INT PRIMARY KEY,
    product_area_name TEXT,
    profit_margin NUMERIC
);

create temp table transactions_temp (
    customer_id NUMERIC NOT NULL,
    transaction_date DATE,
    transaction_id NUMERIC NOT NULL,
    product_area_id INT,
    num_items INT,
    sales_cost NUMERIC
);

-- copy data into created tables
copy delivery_club_campaign_temp(customer_id,campaign_name,campaign_date,mailer_type,signup_flag)
from '/infile/campaign_data.csv'
delimiter ','
csv header;

copy customers(customer_id,distance_from_store,gender,credit_score)
from '/infile/customer_details.csv'
delimiter ','
csv header;

copy loyalty_scores(customer_id,customer_loyalty_score)
from '/infile/loyalty_scores.csv'
delimiter ','
csv header;

copy product_areas(product_area_id,product_area_name,profit_margin)
from '/infile/product_areas.csv'
delimiter ','
csv header;

copy transactions_temp(customer_id,transaction_date,transaction_id,product_area_id,num_items,sales_cost)
from '/infile/transactions.csv'
delimiter ','
csv header;


-- model the data
drop table if exists customer_transactions; 
create table customer_transactions as
    select distinct
        transaction_id,
        customer_id
    from transactions_temp
;

alter table customer_transactions 
    add constraint fk_transaction_customer_id 
    foreign key (customer_id)
    references customers (customer_id),
    add constraint fk_transaction_id 
    foreign key (transaction_id)
    references transaction_details (transaction_id)
    
;

drop table if exists transaction_details; 
create table transaction_details ( 
    transaction_id NUMERIC NOT NULL,
    transaction_date DATE,
    line_number INT NOT NULL,
    product_area_id INT,
    num_items INT,
    sales_cost NUMERIC,
    primary key(transaction_id, line_number),
    constraint fk_product_area
        foreign key (product_area_id)
            references product_areas (product_area_id)
);

insert into transaction_details (transaction_id, transaction_date, line_number, product_area_id, num_items, sales_cost)
    select
        transaction_id,
        transaction_date,
        row_number() over (partition by transaction_id) line_number,
        product_area_id,
        num_items,
        sales_cost
    from transactions_temp
;

drop table if exists campaigns cascade; 
create table campaigns (
    campaign_id serial primary key,
    campaign_name text,
    campaign_date date
);

insert into campaigns (campaign_name, campaign_date)
    select distinct 
        campaign_name, 
        campaign_date 
    from delivery_club_campaign_temp
;

drop table if exists customer_campaign; 
create table customer_campaign as
select
    dc.customer_id,
    c.campaign_id,
    dc.mailer_type,
    dc.signup_flag
from delivery_club_campaign_temp dc
left join campaigns c on dc.campaign_name = c.campaign_name
    and dc.campaign_date = c.campaign_date
;

alter table customer_campaign
    add constraint fk_campaign_id 
    foreign key (campaign_id)
    references campaigns (campaign_id),
    add constraint fk_campaign_customer_id 
    foreign key (customer_id)
    references customers (customer_id)
;

create or replace view transactions_by_customer as
select
    ct.customer_id,
    sum(t.sales_cost) as total_sales,
    sum(t.num_items) as total_items,
    count(distinct ct.transaction_id) as transaction_count,
    sum(t.sales_cost) / count(distinct ct.transaction_id) average_basket_value,  
    count(distinct t.product_area_id) as product_area_count
from groceries.transaction_details t
left join groceries.customer_transactions ct on t.transaction_id = ct.transaction_id
group by ct.customer_id
;

create or replace view loyalty_score_regression_data as 
with lscores as (
select 
    customer_id,
    row_number() over (partition by customer_id order by score_datetime desc) dtt_rnk,
    customer_loyalty_score
from groceries.loyalty_scores)
select
    c.customer_id,
    l.customer_loyalty_score,
    c.distance_from_store,
    c.gender,
    c.credit_score,
    tt.total_sales,
    tt.total_items,
    tt.transaction_count,
    tt.average_basket_value,  
    tt.product_area_count
from groceries.customers c
left join groceries.transactions_by_customer tt on c.customer_id = tt.customer_id
left join lscores l on c.customer_id = l.customer_id
where l.dtt_rnk = 1
;

create or replace view signup_classification_data as 
select
    c.customer_id,
    cc.signup_flag,
    c.distance_from_store,
    c.gender,
    c.credit_score,
    tt.total_sales,
    tt.total_items,
    tt.transaction_count,
    tt.average_basket_value,  
    tt.product_area_count
from groceries.customers c
left join groceries.transactions_by_customer tt on c.customer_id = tt.customer_id
left join groceries.customer_campaign cc on c.customer_id = cc.customer_id
;