# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 06:00:45 2021

@author: sheld
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import yfinance as yf
from yahoofinancials import YahooFinancials
import datetime
from datetime import datetime,timedelta
from dateutil.relativedelta import relativedelta
import numpy as np
import numpy_financial as npf
import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt
# import numpy as np
# from data.create_data import create_table

#data_dir = 'C:/Users/sheld/OneDrive/Documents/PortfolioOpt/Data/'
data_dir = './Data/'
CAPITAL_CITIES=['Sydney','Melbourne','Brisbane','Adelaide','Perth','Canberra','Hobart','Darwin']
PROPERTY_TYPE=['Established House','Attached Dwelling']

st.title('Shares v.s. Property')
st.write('Shares or Property? which is the better investment for a long term capital growth?\
         It is a prolonged hot debate...Hopefully we can unfold some insights here.')
initial_cap = st.number_input(label='Initial capital to invest (Min. $30,000)',format='%d',min_value=30000,step=1000,value=100000)
invest_year = st.slider(label='Year when you start investing',min_value=2002,max_value=2020,value=2017,step=1)
st.subheader('Property')
location = st.selectbox('Location',CAPITAL_CITIES,0)
# property_type = st.selectbox('Property Type',PROPERTY_TYPE,1)
mortgage_term = st.slider('Mortgage Term',min_value=10,max_value=30,value=30,step=1)
purchase_price = st.slider('Property Purchase Price (Assume at least 20% deposit)', min_value=initial_cap,max_value=round(initial_cap/1000/0.2)*1000,value=round(initial_cap/1000/0.2)*1000,step=1000)
stamp_duty = st.number_input('Stamp Duty (% of the purchase price)',min_value=0,max_value=10,value=4,step=1)
maintenance = st.number_input('Ongoing Maintenance $ per month',value=100,step=100)
rent = st.number_input('Initial Rental Income $ per week (Assume 1.5% increase per year)',value=300,step=10)
property_management = st.number_input('Property Management Fee (% of the total rent)',min_value=1,max_value=10,value=5,step=1)
sales_agent_fee = st.number_input('Sales Agent Fee (% of the sales price)',min_value=1,max_value=5,value=2,step=1)
legal_fee = st.number_input('Legal Cost',value=3000)
banking_fee = st.number_input('Banking Fees $ per year',value=399)
councile_rate = st.number_input('Council Rates $ per year',value=1600)
water_rate = st.number_input('Water Rate $ per month',value=80)
df_property = pd.read_csv(data_dir+'house_price.csv')
df_property.Date = pd.to_datetime(df_property.Date,format='%d/%m/%Y')
# df_property.set_index('Date',inplace=True)
df_property[CAPITAL_CITIES] = df_property[CAPITAL_CITIES]*1000
# date_now = date.today().strftime('%Y-%m-%d')
# df_property = df_property[df_property.index.month==12]
# df_property.index = df_property.index.year 
df_sub = pd.DataFrame(df_property[['Date',location]][df_property.Date.dt.year>=invest_year])
df_sub.reset_index(inplace=True,drop=True)
ratio = purchase_price/df_sub.loc[0,location]
df_sub['House_Price'] = df_sub[location]*ratio

df_rate = pd.read_csv(data_dir+'interest_rate.csv')
df_rate.Date = pd.to_datetime(df_rate.Date,format='%d/%m/%Y')
#df_rate.set_index('Date',inplace=True)

#Mortgage calculation 
start_date = datetime(invest_year, 1, 1)
rng = pd.date_range(start_date, periods=mortgage_term * 12, freq='MS')
rng.name = "Date"
df_mortgage = pd.DataFrame(index=rng, columns=['Payment', 'Principal Paid', 'Interest Paid', 'Ending Balance'], dtype='float')
df_mortgage.reset_index(inplace=True)
df_mortgage.index += 1
df_mortgage.index.name = "Period"
df_mortgage = df_mortgage.merge(df_rate[['Date','Interest_Rate_Discounted']],left_on='Date',right_on='Date',how='left')
mortgage = purchase_price - initial_cap
df_mortgage.loc[0,'Payment'] = -1*npf.pmt(df_mortgage.loc[0,'Interest_Rate_Discounted']/100/12,mortgage_term*12,mortgage)
df_mortgage.loc[0,'Principal Paid'] = -1*npf.ppmt(df_mortgage.loc[0,'Interest_Rate_Discounted']/100/12,1,mortgage_term*12,mortgage)
df_mortgage.loc[0,'Interest Paid'] = -1*npf.ipmt(df_mortgage.loc[0,'Interest_Rate_Discounted']/100/12,1,mortgage_term*12,mortgage)
df_mortgage.loc[0,'Ending Balance'] = mortgage - df_mortgage.loc[0,'Principal Paid']
df_mortgage.loc[0,'Accumulative Interest'] = df_mortgage.loc[0,'Interest Paid']
for indx in df_mortgage.index[1:]:
    df_mortgage.loc[indx,'Payment'] = -1*npf.pmt(df_mortgage.loc[indx,'Interest_Rate_Discounted']/100/12,mortgage_term*12,df_mortgage.loc[indx-1,'Ending Balance'])
    df_mortgage.loc[indx,'Principal Paid'] = -1*npf.ppmt(df_mortgage.loc[indx,'Interest_Rate_Discounted']/100/12,1,mortgage_term*12,df_mortgage.loc[indx-1,'Ending Balance'])
    df_mortgage.loc[indx,'Interest Paid'] = -1*npf.ipmt(df_mortgage.loc[indx,'Interest_Rate_Discounted']/100/12,1,mortgage_term*12,df_mortgage.loc[indx-1,'Ending Balance'])
    df_mortgage.loc[indx,'Accumulative Interest'] = df_mortgage.loc[indx-1,'Accumulative Interest'] + df_mortgage.loc[indx,'Interest Paid']
    df_mortgage.loc[indx,'Ending Balance'] =  df_mortgage.loc[indx-1,'Ending Balance'] - df_mortgage.loc[indx,'Principal Paid']
#Merge with property price 
df_sub = df_sub.merge(df_mortgage,left_on='Date',right_on='Date',how='left')

#Calculate the profit 
df_sub['Invest_Year'] = datetime(invest_year,1,1)
df_sub['time_diff'] = (df_sub['Date'] - df_sub['Invest_Year']).astype('timedelta64[M]')
df_sub['Cost'] = df_sub['Accumulative Interest']+stamp_duty/100*df_sub['House_Price']+maintenance*df_sub['time_diff']+rent*df_sub['time_diff']*4*property_management/100+sales_agent_fee/100*df_sub['House_Price']+legal_fee+banking_fee*df_sub['time_diff']/12+councile_rate*df_sub['time_diff']/12+water_rate*df_sub['time_diff']
df_sub['Rental'] = rent*(1.015**df_sub.index)
df_sub.loc[0,'Accumulative Rent'] = df_sub.loc[0,'Rental']*52/12*df_sub.loc[0,'time_diff']
for indx in df_sub.index[1:]:
    df_sub.loc[indx,'Accumulative Rent'] = df_sub.loc[indx-1,'Accumulative Rent'] + df_sub.loc[indx,'Rental']*52/12*(df_sub.loc[indx,'time_diff']-df_sub.loc[indx-1,'time_diff'])

df_sub['Profit'] = df_sub['House_Price']+df_sub['Accumulative Rent']-df_sub['Ending Balance']-initial_cap-df_sub['Cost']
# time_diff = relativedelta(df_sub.loc[df_sub.shape[0]-1,'Date'],datetime(invest_year,1,1))
# total_month = time_diff.years*12+time_diff.months
# final_price = df_sub.loc[df_sub.shape[0]-1,'House_Price']
# total_cost = sum(df_sub['Interest Paid'])+stamp_duty/100*purchase_price+maintenance*total_month+rent*total_month*4*property_management/100+sales_agent_fee/100*purchase_price+legal_fee+banking_fee*time_diff.years+councile_rate*time_diff.years+water_rate*total_month
# rental = rent*52/12*total_month
# gross_capital_gain_property = final_price-df_sub.loc[df_sub.shape[0]-1,'Ending Balance']-initial_cap
# net_capital_gain_property = gross_capital_gain_property - total_cost

# =============================================================================
# Shares
# =============================================================================
st.subheader('Shares')
brokage_fee = st.number_input('Brokage fee for shares $',value=30,step=1)
df_nasdaq = pd.read_csv(data_dir+'QQQ.csv')
df_nasdaq['Date'] = pd.to_datetime(df_nasdaq.Date)
df_nasdaq = df_nasdaq.rename(columns={'Adj Close':'Nasdaq Close'})
df_sub = df_sub.merge(df_nasdaq[['Date','Nasdaq Close']],how='left',left_on='Date',right_on='Date')
df_sub['Profit_Nasdaq'] = np.floor(initial_cap/df_sub.loc[0,'Nasdaq Close'])*(df_sub['Nasdaq Close'] - df_sub.loc[0,'Nasdaq Close'])-brokage_fee

df_sp500 = pd.read_csv(data_dir+'SPY.csv')
df_sp500['Date'] = pd.to_datetime(df_sp500.Date)
df_sp500 = df_sp500.rename(columns={'Adj Close':'sp500 Close'})
df_sub = df_sub.merge(df_sp500[['Date','sp500 Close']],how='left',left_on='Date',right_on='Date')
df_sub['Profit_Sp500'] = np.floor(initial_cap/df_sub.loc[0,'sp500 Close'])*(df_sub['sp500 Close'] - df_sub.loc[0,'sp500 Close'])-brokage_fee
# =============================================================================
# Plot the profit 
# =============================================================================
fig = plt.figure(figsize=(10, 7))
plt.plot(df_sub['Date'],df_sub['Profit'],label='Property')
plt.plot(df_sub['Date'], df_sub['Profit_Nasdaq'],color='red'
         #,linewidth=1.0,linestyle='--'
         ,label='Nasdaq'
         )
plt.plot(df_sub['Date'], df_sub['Profit_Sp500'],color='green'
         #,linewidth=1.0,linestyle='--'
         ,label='S&P500'
         )
plt.title('Investment Profit over Time')
plt.xlabel('Date')
plt.ylabel('Profit$')
plt.legend(labelspacing=0.8)
# for i, txt in enumerate(stocks_to_opt):
#     plt.annotate(txt, (an_vol[i],an_rt[i]), xytext=(10,0), textcoords='offset points')

st.pyplot(fig)