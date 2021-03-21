# -*- coding: utf-8 -*-
"""
Created on Sun Mar 21 15:17:58 2021

@author: SLIN
"""

from datetime import date 
import pandas as pd
import gc

today = date.today()
report_date = 'Dec-2020'   #This needs to be latest date of ABS(Australia Bureau of Statistics) report. 

# =============================================================================
# Download the inflation data from ABS table 640109
# =============================================================================
url = 'https://www.abs.gov.au/statistics/economy/price-indexes-and-inflation/consumer-price-index-australia/' + report_date + '/640109.xls'
xls = pd.ExcelFile(url) #use r before absolute file path 
sheetX = xls.parse(1,header=None,index_col=None) #2 is the sheet number+1 thus if the file has only 1 sheet write 0 in paranthesis
df_inflation = sheetX.iloc[9:]
df_inflation = df_inflation.rename(columns=df_inflation.iloc[0]).drop(df_inflation.index[0])

for index in range(2,7):
    sheetX = xls.parse(index,header=None,index_col=None) #2 is the sheet number+1 thus if the file has only 1 sheet write 0 in paranthesis
    sheetX = sheetX.iloc[9:,]
    sheetX = sheetX.rename(columns=sheetX.iloc[0]).drop(sheetX.index[0]).iloc[:,1:]
    df_inflation = pd.concat([df_inflation,sheetX],axis=1)
del sheetX,xls
gc.collect()

Columns = ['Series ID','A2331840C','A2331845R','A2331850J','A2331855V','A2331860L','A2331865X','A2331870T','A2331875C']
New_Columns = ['Date','Sydney','Melbourne','Brisbane','Adelaide','Perth','Hobart','Darwin','Canberra']

df_inflation = df_inflation[Columns]
df_inflation.columns = New_Columns
df_inflation=df_inflation[df_inflation.Date>=datetime(2002,3,1)]
df_inflation.reset_index(inplace=True,drop=True)

df_inflation.to_csv('inflation_data.csv',index=False)

# =============================================================================
# Download the property price data from ABS table xxxx
# =============================================================================

