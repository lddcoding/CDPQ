import streamlit as st
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import pandas_datareader as web
import datetime
from datetime import date

def annualized_return(Df, nb_of_year):
    
    Df = Df[::-1].reset_index()
    last_value = Df['Adj Close'][0]
    first_value = Df['Adj Close'].tail(1)
  
    rate_of_return = (last_value - first_value) / first_value
    an_return = ((1 + rate_of_return)**(1/nb_of_year)) - 1
    an_return = an_return * 100
    df = pd.DataFrame(an_return)
    df.columns = [f'{stock.upper()} Annualized Return on {nb_of_year} years']
    df.index = ['Row_1']
    return df

try:
    nb_of_year = st.number_input('How many year span: ', min_value=1, max_value=15, step=1)
    start = datetime.datetime.now() - datetime.timedelta(days=nb_of_year*365)
    end = date.today()

    stock = st.text_input("Stock: ")
    dataframe = web.DataReader(f'{stock}', "yahoo", start, end)

    st.dataframe(annualized_return(dataframe, nb_of_year))
except Exception as e: 
    st.write(e)
    
