# -*- coding: utf-8 -*-
"""
Created on Mon Mar  1 06:00:45 2021

@author: sheld
"""
import os
import streamlit as st
import pandas as pd
# import plotly.express as px
# import yfinance as yf
# from yahoofinancials import YahooFinancials
import datetime
from datetime import datetime,timedelta
from dateutil.relativedelta import relativedelta
import numpy as np
import numpy_financial as npf
import matplotlib.pyplot as plt
from PIL import Image
import plotly.graph_objects as go
# import matplotlib.pyplot as plt
# import numpy as np
# from data.create_data import create_table

data_dir = os.getcwd()+'/'
CAPITAL_CITIES=['Sydney','Melbourne','Brisbane','Adelaide','Perth','Canberra','Hobart','Darwin']
PROPERTY_TYPE=['House','Apartment/Unit']

st.set_page_config( layout='centered')

st.title('Shares v.s. Property')
st.write('Shares or Property? Which is better investment for a long term capital growth?\
         It is a prolonged debate...Hopefully we can unfold some insights here.')

col1, col2 = st.beta_columns([3, 3])
with col1:
    image1 = Image.open(data_dir+'wall street.jpg')
    st.image(image1,use_column_width=True)
with col2:
    image2 = Image.open(data_dir+'house.jpg')
    st.image(image2,use_column_width=True)

initial_cap = st.number_input(label='Initial capital to invest (Min. $30,000)',format='%d',min_value=30000,step=1000,value=50000)
invest_year = st.slider(label='Year when you start investing',min_value=2002,max_value=2020,value=2002,step=1)
st.subheader('Property')
stamp_duty = st.number_input('Stamp Duty (% of the purchase price)',min_value=0,max_value=10,value=4,step=1)
purchase_price = st.slider('Property Purchase Price (Assume at least 20% deposit after stamp duty)', min_value=round(initial_cap/(1+stamp_duty/100)/1000)*1000,max_value=round(initial_cap/(0.2+stamp_duty/100)/1000)*1000,value=round(initial_cap/(0.2+stamp_duty/100)/1000)*1000,step=1000)
property_type = st.radio('Property Type',PROPERTY_TYPE)
location = st.selectbox('Location',CAPITAL_CITIES,0)

with st.beta_expander("More Settings"):
    col1, col2 = st.beta_columns([3, 3])
    with col1:
        # property_type = st.selectbox('Property Type',PROPERTY_TYPE,1)
        mortgage_term = st.slider('Mortgage Term',min_value=10,max_value=30,value=30,step=1)
        maintenance = st.number_input('Ongoing Maintenance $ per month',value=50,step=10)
        rent = st.number_input('Initial Rental Income $ per week (Default 3.5% rental yield)',value=int(purchase_price*0.035/52),step=10)
        property_management = st.number_input('Property Management Fee (% of the total rent)',min_value=1,max_value=10,value=5,step=1)
        if property_type == 'Apartment/Unit':
            strata = st.number_input('Initial Strata Levy $per quarter (Assume 1% increase per year)',value=300,step=10)
        else:
            strata = 0
    with col2:
        
        sales_agent_fee = st.number_input('Sales Agent Fee (% of the sales price)',min_value=1,max_value=5,value=2,step=1)
        legal_fee = st.number_input('Legal Cost',value=3000)
        banking_fee = st.number_input('Banking Fees $ per year',value=399)
        councile_rate = st.number_input('Council Rates $ per year',value=1600)
        water_rate = st.number_input('Water Rate $ per month',value=80)

if property_type=='House':
    df_property = pd.read_csv(data_dir+'property_price_house.csv')
else:
    df_property = pd.read_csv(data_dir+'property_price_apartment.csv')
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

#Load rent inflation data 
df_inflation = pd.read_csv(data_dir+'inflation_data.csv')
df_inflation = df_inflation[['Date',location]]
df_inflation.rename(columns={location:'Rent_Inflation'},inplace=True)
df_inflation.Date = pd.to_datetime(df_inflation.Date)
df_sub = df_sub.merge(df_inflation,left_on='Date',right_on='Date',how='left')

#Calculate the profit 
df_sub['Invest_Year'] = datetime(invest_year,1,1)
df_sub['time_diff'] = (df_sub['Date'] - df_sub['Invest_Year']).astype('timedelta64[M]')
df_sub['Cost'] = df_sub['Accumulative Interest']+stamp_duty/100*df_sub['House_Price']+maintenance*df_sub['time_diff']+rent*df_sub['time_diff']*4*property_management/100+sales_agent_fee/100*df_sub['House_Price']+legal_fee+banking_fee*df_sub['time_diff']/12+councile_rate*df_sub['time_diff']/12+water_rate*df_sub['time_diff']
# df_sub['Rental'] = rent*((1+(0.015/12*3))**df_sub.index)
df_sub['Strata'] = strata*((1+(0.01/12*3))**df_sub.index)
df_sub.loc[0,'Rental'] = rent  
df_sub.loc[0,'Accumulative Rent'] = df_sub.loc[0,'Rental']*52/12*df_sub.loc[0,'time_diff']
df_sub.loc[0,'Accumulative Strata'] = df_sub.loc[0,'Strata']*4/12*df_sub.loc[0,'time_diff']
for indx in df_sub.index[1:]:
    df_sub.loc[indx,'Rental'] = df_sub.loc[indx-1,'Rental']*(df_sub.loc[indx,'Rent_Inflation']/100+1)
    df_sub.loc[indx,'Accumulative Rent'] = df_sub.loc[indx-1,'Accumulative Rent'] + df_sub.loc[indx,'Rental']*52/12*(df_sub.loc[indx,'time_diff']-df_sub.loc[indx-1,'time_diff'])
    df_sub.loc[indx,'Accumulative Strata'] = df_sub.loc[indx-1,'Accumulative Strata'] + df_sub.loc[indx,'Strata']*4/12*(df_sub.loc[indx,'time_diff']-df_sub.loc[indx-1,'time_diff'])
df_sub['Profit'] = df_sub['House_Price']+df_sub['Accumulative Rent']-df_sub['Ending Balance']-initial_cap-df_sub['Cost']-df_sub['Accumulative Strata']
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
brokerage_fee1 = st.number_input('Brokerage fee for value up to $25,000 in $',value=30,step=1)
brokerage_fee2 = st.number_input('Brokerage fee for value over $25,000 in %',value=0.12,step=0.01)
if initial_cap<=25000:
    brokerage_fee = brokerage_fee1
else: 
    brokerage_fee = brokerage_fee1+(initial_cap-25000)*brokerage_fee2/100

df_nasdaq = pd.read_csv(data_dir+'QQQ.csv')
df_nasdaq['Date'] = pd.to_datetime(df_nasdaq.Date)
df_nasdaq = df_nasdaq.rename(columns={'Adj Close':'Nasdaq Close'})
df_sub = df_sub.merge(df_nasdaq[['Date','Nasdaq Close']],how='left',left_on='Date',right_on='Date')
df_sub['Profit_Nasdaq'] = np.floor(initial_cap/df_sub.loc[0,'Nasdaq Close'])*(df_sub['Nasdaq Close'] - df_sub.loc[0,'Nasdaq Close'])-brokerage_fee1-brokerage_fee2

df_sp500 = pd.read_csv(data_dir+'SPY.csv')
df_sp500['Date'] = pd.to_datetime(df_sp500.Date)
df_sp500 = df_sp500.rename(columns={'Adj Close':'sp500 Close'})
df_sub = df_sub.merge(df_sp500[['Date','sp500 Close']],how='left',left_on='Date',right_on='Date')
df_sub['Profit_Sp500'] = np.floor(initial_cap/df_sub.loc[0,'sp500 Close'])*(df_sub['sp500 Close'] - df_sub.loc[0,'sp500 Close'])-brokerage_fee1-brokerage_fee2

# =============================================================================
# Plot the profit 
# =============================================================================

if st.button('Display Results!'):

    # =============================================================================
    # Calculate the annulised return 
    # =============================================================================
    df_sub['Profit_Property_Rate'] = (((df_sub['Profit']+initial_cap)/initial_cap)**(1/df_sub['time_diff'])-1)*12
    df_sub['Profit_Sp500_Rate'] = (((df_sub['Profit_Sp500']+initial_cap)/initial_cap)**(1/df_sub['time_diff'])-1)*12
    df_sub['Profit_Nasdaq_Rate'] = (((df_sub['Profit_Nasdaq']+initial_cap)/initial_cap)**(1/df_sub['time_diff'])-1)*12
    
    df_annulised_return = pd.DataFrame({'Property':[df_sub.iloc[-1]['Profit_Property_Rate']],
                                        'S&P 500':[df_sub.iloc[-1]['Profit_Sp500_Rate']],
                                        'Nasdaq':[df_sub.iloc[-1]['Profit_Nasdaq_Rate']]})
    
    
    st.write('Annulised Return Since Inception')
    st.write(df_annulised_return.style.format({
    'Property': '{:,.2%}'.format,
    'S&P 500': '{:,.2%}'.format,
    'Nasdaq': '{:,.2%}'.format,
    }))
    
    # st.line.chart()

    fig1 = go.Figure()
    fig1.add_trace(go.Scatter(x=df_sub['Date'], y=df_sub['Profit'],
                        mode='lines',
                        name='Property'))
    fig1.add_trace(go.Scatter(x=df_sub['Date'], y=df_sub['Profit_Sp500'],
                        mode='lines',
                        name='S&P 500'))
    fig1.add_trace(go.Scatter(x=df_sub['Date'], y=df_sub['Profit_Nasdaq'],
                        mode='lines', name='Nasdaq'))
    fig1.update_layout(
            autosize=False,
            # width=300,
            height=450,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            title={
            'text': 'Investment Profit over Time',
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
                margin=dict(
                l=5,
                r=5,
                b=5,
                t=70,
                pad=0
            ),
        # title='Investment Profit over Time'
        # font_family="Courier New",
        # font_color="blue",
        # title_font_family="Times New Roman",
        # title_font_color="red",
        # legend_title_font_color="green"
    )    
    
    fig2 = go.Figure()
    # fig2.add_trace(go.Scatter(x=df_sub['Date'], y=df_sub[location],
    #                     mode='lines',line = dict(color='firebrick', width=2 ,dash='dot'),
    #                     name='Median Price'))
    fig2.add_trace(go.Scatter(x=df_sub['Date'], y=df_sub['House_Price'],
                        mode='lines',line = dict(color='royalblue', width=2),
                        name='Estimated Price'))
    fig2.update_layout(
                 legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
        title={
            'text': 'Property Price Over Time',
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
             margin=dict(
            l=5,
            r=5,
            b=5,
            t=70,
            pad=0
        ),
    
    )
    
    
    
    # fig3 = go.Figure()
    # fig3.add_trace(go.Scatter(x=df_sub['Date'], y=df_sub['Profit_Property_Rate'],
    #                     mode='lines',
    #                     name='Property'))
    # fig3.add_trace(go.Scatter(x=df_sub['Date'], y=df_sub['Profit_Sp500_Rate'],
    #                     mode='lines',
    #                     name='S&P 500'))
    # fig3.add_trace(go.Scatter(x=df_sub['Date'], y=df_sub['Profit_Nasdaq_Rate'],
    #                     mode='lines', name='Nasdaq'))
    # fig3.update_layout(
    #         title={
    #         'text': 'Annualised Rate of Return %',
    #         'y':0.9,
    #         'x':0.5,
    #         'xanchor': 'center',
    #         'yanchor': 'top'},
    #         yaxis_tickformat = '%',
    #         yaxis_range = [-0.5,0.5]
             
    #     # title='Investment Profit over Time'
    #     # font_family="Courier New",
    #     # font_color="blue",
    #     # title_font_family="Times New Roman",
    #     # title_font_color="red",
    #     # legend_title_font_color="green"
    # )
    
    fig4 = go.Figure()
    # fig2.add_trace(go.Scatter(x=df_sub['Date'], y=df_sub[location],
    #                     mode='lines',line = dict(color='firebrick', width=2 ,dash='dot'),
    #                     name='Median Price'))
    fig4.add_trace(go.Scatter(x=df_sub['Date'], y=df_sub['Nasdaq Close'],
                        mode='lines',line = dict(color='green', width=2),
                        name='Estimated Price'))
    fig4.update_layout(
                 legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
        title={
            'text': 'Nasdaq ETF Adjusted Close Over Time',
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
                 margin=dict(
                l=5,
                r=5,
                b=5,
                t=70,
                pad=0
            ),
    
    )
    
    fig5 = go.Figure()
    # fig2.add_trace(go.Scatter(x=df_sub['Date'], y=df_sub[location],
    #                     mode='lines',line = dict(color='firebrick', width=2 ,dash='dot'),
    #                     name='Median Price'))
    fig5.add_trace(go.Scatter(x=df_sub['Date'], y=df_sub['sp500 Close'],
                        mode='lines',line = dict(color='red', width=2),
                        name='Estimated Price'))
    fig5.update_layout(
                 legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
        title={
            'text': 'S&P500 ETF Adjusted Close Over Time',
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
                 margin=dict(
                l=5,
                r=5,
                b=5,
                t=70,
                pad=0
            ),
    
    )
    
    fig6 = go.Figure()
    # fig2.add_trace(go.Scatter(x=df_sub['Date'], y=df_sub[location],
    #                     mode='lines',line = dict(color='firebrick', width=2 ,dash='dot'),
    #                     name='Median Price'))
    fig6.add_trace(go.Scatter(x=df_sub['Date'], y=df_sub['Rental'],
                        mode='lines',line = dict(color='orange', width=2),
                        name='Estimated Price'))
    fig6.update_layout(
                 legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
        title={
            'text': 'Rent/week (adjusted by inflation) Over Time',
            'y':0.9,
            'x':0.5,
            'xanchor': 'center',
            'yanchor': 'top'},
                 margin=dict(
                l=5,
                r=5,
                b=5,
                t=70,
                pad=0
            ),
    
    )
    
    st.plotly_chart(fig1,use_container_width=True)
    st.plotly_chart(fig2,use_container_width=True)
    st.plotly_chart(fig4,use_container_width=True)
    st.plotly_chart(fig5,use_container_width=True)
    st.plotly_chart(fig6,use_container_width=True)


# st.write(df_sub)

