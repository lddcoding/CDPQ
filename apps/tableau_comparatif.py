import streamlit as st
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import pandas_datareader as web
import datetime
from datetime import date
from multiapp import MultiApp
from apps import tableau_comparatif, data_stats 
import base64
import time


#------------------------------------------------------------------Functions----------------------------------------------------------------------------------------


def annualized_return(Df, nb_of_year, stock):
    
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


def get_table_download_link_csv(df, name_file):
    #csv = df.to_csv(index=False)
    csv = df.to_csv().encode()
    #b64 = base64.b64encode(csv.encode()).decode() 
    b64 = base64.b64encode(csv).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{name_file}.csv" target="_blank">Download csv file</a>'
    return href


def volatility_per_month(stock, nb_of_year):
    
    start = datetime.datetime.now() - datetime.timedelta(days=nb_of_year*365)
    end = date.today()
    dataframe = web.DataReader(f'{stock}', "yahoo", start, end)
    
    df_series = dataframe['Adj Close'].resample('M').ffill()
    df = pd.DataFrame(df_series)
    returns = df["Adj Close"].pct_change()
    volatility = returns.rolling(window=2).std()*np.sqrt(252) # We choose a default window of 30 for the number of day in a month
    df = pd.DataFrame(volatility)
    df = df.rename(columns={"Adj Close":"Volatility"})
    df.reset_index(inplace=True)
    return df

#------------------------------------------------------------------App----------------------------------------------------------------------------------------


def app():
    try:
        stock1 = st.text_input("Enter your first stock ticker: ")
        stock2 = st.text_input("Enter your second stock ticker: ")

        # nb_of_year = st.number_input('How many year(s) span (1Y to 15Y): ', min_value=1, max_value=15, step=1)
        list_of_years = [1, 3, 5, 10]
        dic_annualized_return = {}

        dropdown = st.selectbox('Analysis Type: ',  ['', 'Calculate Annualized Return', 'Calculate Volatility per Month'], format_func=lambda x: 'Select an option' if x == '' else x)
        if dropdown == ('Calculate Annualized Return'):
            my_bar = st.progress(0)
            y = 0

            try:
        
                for i in list_of_years:

                    start = datetime.datetime.now() - datetime.timedelta(days=i*365)
                    end = date.today()

                    dataframe1 = web.DataReader(f'{stock1}', "yahoo", start, end)
                    dataframe2 = web.DataReader(f'{stock2}', "yahoo", start, end)

                    df1 = annualized_return(dataframe1, i, stock1)
                    df2 = annualized_return(dataframe2, i, stock2)

                    concatcolumns_df = pd.concat([df1, df2], axis = 1)
            
                    dic_annualized_return[i] = [concatcolumns_df.iloc[0][0], concatcolumns_df.iloc[0][1]]
                    df_annualized_return = pd.DataFrame(dic_annualized_return)
                    df_annualized_return = df_annualized_return.transpose()
                    df_annualized_return = df_annualized_return.reset_index(drop=True)
                    df_annualized_return.columns = [stock1, stock2]
                    y += 25
                    my_bar.progress(y)
                    
                df_annualized_return['Years span'] = list_of_years
                st.dataframe(df_annualized_return)
                st.markdown(get_table_download_link_csv(df_annualized_return, 'Annualized Return'), unsafe_allow_html=True)
                st.success('Done') 

            except:
                st.error('Please enter two stocks')
            

        if dropdown == ('Calculate Volatility per Month'):
            try:

                my_bar_2 = st.progress(0)
                df_v1 = volatility_per_month(stock1, 10)
                my_bar_2.progress(20)
                df_v2 = volatility_per_month(stock2, 10)
                my_bar_2.progress(40)

                df_v2.drop(df_v2.columns[0], axis =1, inplace=True)
                df_v1.rename(columns = {df_v1.columns[1] : stock1.upper()}, inplace=True)
                df_v2.rename(columns = {df_v2.columns[0] : stock2.upper()}, inplace=True)
                my_bar_2.progress(60)

                df = pd.concat([df_v1, df_v2], axis = 1)
                df['Date'] = df['Date'].dt.strftime('%Y-%m-%d')
                my_bar_2.progress(80)
            
                st.dataframe(df)
                st.markdown(get_table_download_link_csv(df, 'Volatility Monthly'), unsafe_allow_html=True)
                my_bar_2.progress(100)
                st.success('Done')

            except:
                my_bar_2.progress(0)
                st.error('Please enter two stocks') 

        
        if dropdown == (''):
            st.warning('No option is selected')

                
    except Exception as e: 
        st.write(e)


