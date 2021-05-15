import streamlit as st
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import pandas_datareader as web
import datetime
from datetime import date

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

def volatility(DF):
    df = DF.copy()
    df["daily_ret"] = DF["Adj Close"].pct_change()
    vol = df["daily_ret"].std() * np.sqrt(252)
    return vol

def volatility_per_day(dataframe):
    df_series = dataframe['Adj Close'].resample('d').ffill()
    df = pd.DataFrame(df_series)
    returns = df["Adj Close"].pct_change()
    volatility = returns.rolling(window=30).std()*np.sqrt(252) # We choose a default window of 30 for the number of day in a month
    df = pd.DataFrame(volatility)
    df = df.rename(columns={"Adj Close":"Volatility"})
    df.reset_index(inplace=True)
    df
    
    plt.figure(figsize=(16,8))
    plt.plot(df['Date'], df['Volatility'])
    plt.xlabel('Years')
    plt.ylabel('Volatility')
    plt.title('Volatility over 10 years in monthly frequences')
    plt.show()

def plot_your_volatility_per_month(df, stock):
    df_series = df['Adj Close'].resample('M').ffill()
    df = pd.DataFrame(df_series)
    df["Rate of Returns"] = df["Adj Close"].pct_change()
    volatility = df["Rate of Returns"].std() * np.sqrt(252) 
    volatility = round(volatility, 4)*100
    
    fig = plt.figure(figsize = (14,6))
    ax = fig.gca()
    df['Rate of Returns'].hist(ax=ax, bins=50, alpha=0.6, edgecolor='black', color='blue')
    ax.grid(False)
    ax.set_xlabel('Monthly Return')
    ax.set_ylabel('Frequence')
    ax.set_title(f'{stock.upper()} volatility: {volatility}%')

def two_graph_plot(ticker1, ticker2):
    
    #Initalizing time frame
    start = datetime.datetime.now() - datetime.timedelta(days=10*365)
    end = date.today()

    #Initializing the ticker
    stock1 = ticker1
    stock2 = ticker2
    df1 = web.DataReader(f'{stock1}', "yahoo", start, end)
    df2 = web.DataReader(f'{stock2}', "yahoo", start, end)

    #Rendement cumulatif
    for stock_df in (df1,df2):
        stock_df["Normed Return"] = stock_df["Adj Close"]/stock_df.iloc[0]["Adj Close"]

    #Setting up the graph
    fig = plt.figure(figsize=(10,10))
    ax1 = fig.add_subplot(2, 1, 1)
    ax2 = fig.add_subplot(2, 1, 2)

    ax1.plot(df1['Normed Return'], c='blue', label=f'{stock1}')
    ax2.plot(df2['Normed Return'], c='green', label=f'{stock2}')
    
    ax1.set_xlabel('Years')
    ax2.set_xlabel('Years')
    
    ax1.set_ylabel('Cumulatize Return')
    ax2.set_ylabel('Cumulatize Return')
    
    ax1.set_title(f'{stock1.upper()} price over 10 years')
    ax2.set_title(f'{stock2.upper()} price over 10 years')

    plt.legend(loc='upper left')
    plt.show()

def one_graph_plot(ticker1, ticker2):
    
    #Initalizing time frame
    start = datetime.datetime.now() - datetime.timedelta(days=10*365)
    end = date.today()
    
    #Initializing the ticker
    stock1 = ticker1
    stock2 = ticker2
    df1 = web.DataReader(f'{stock1}', "yahoo", start, end)
    df2 = web.DataReader(f'{stock2}', "yahoo", start, end)

    #Rendement cumulatif
    for stock_df in (df1,df2):
        stock_df["Normed Return"] = stock_df["Adj Close"]/stock_df.iloc[0]["Adj Close"]

    #Setting up the graph
    fig = plt.figure(figsize=(10,6))

    plt.plot(df1['Normed Return'], c='blue', label=stock1)
    plt.plot(df2['Normed Return'], c='green', label=stock2)

    plt.title(f'{stock1.upper()} vs {stock2.upper()} price over {nb_of_year}')
    plt.xlabel('Years')
    plt.ylabel('Cumulative Return')

    plt.legend(loc='upper left')
    plt.show()

def rendement_annualiser(dataframe, nb_of_year):
    annualized_returns = dataframe['Adj Close'].resample('Y').ffill().pct_change()
    list_of_return = []
    for i in annualized_returns:
        an_return = ((1 + i)**(1/nb_of_year)) - 1
        an_return = an_return * 100
        list_of_return.append(an_return)

    df1 = pd.DataFrame(list_of_return, columns = [f'Annualized Return on {nb_of_year} years (%)'])
    df2 = pd.DataFrame(annualized_returns)
    df2.reset_index(inplace = True)
    df = pd.concat([df1, df2], axis = 1)
    df['Rate of Return (%)'] = df['Adj Close'] * 100
    df.drop(labels='Adj Close', axis="columns", inplace=True)
    df = df[['Date', 'Rate of Return (%)', f'Annualized Return on {nb_of_year} years (%)']]
    
    plt.figure(figsize=(10,5))
    plt.plot(df['Date'], df[f'Annualized Return on {nb_of_year} years (%)'], color='green')
    plt.xlabel('Years')
    plt.ylabel(f'Annualized Return on {nb_of_year} Return')
    plt.title('Annualized Return since 10 years ago')
    plt.show()
    
    return df

def rendement_mensuel(dataframe, nb_of_year):
    monthly_returns = dataframe['Adj Close'].resample('M').ffill().pct_change()
    list_of_return = []
    for i in monthly_returns:
        an_return = ((1 + i)**(1/nb_of_year)) - 1  # Not necessary if it's on a 1 years time frame
        an_return = an_return * 100
        list_of_return.append(an_return)

    df1 = pd.DataFrame(list_of_return, columns = [f'Annualized Return on {nb_of_year} years'])
    df2 = pd.DataFrame(monthly_returns)
    df2.reset_index(inplace = True)
    df = pd.concat([df1, df2], axis = 1)
    df['Monthly Return'] = df['Adj Close'] * 100
    df.drop(labels='Adj Close', axis='columns', inplace=True)
    df = df[['Date', 'Monthly Return', f'Annualized Return on {nb_of_year} years']]
    
    plt.figure(figsize=(10,5))
    plt.plot(df['Date'], df['Monthly Return'], color='purple')
    plt.xlabel('Years (Month Frequency)')
    plt.ylabel('Monthly Return')
    plt.title('Monthly Return since 10 years ago')
    plt.show()
    
    
    return df

def volatility_for_10y(dataframe, stock):
    TRADING_DAYS = 252
    df = dataframe.copy()
    returns = np.log(df['Adj Close']/df['Adj Close'].shift(1))
    returns.fillna(0, inplace=True)
    volatility = returns.rolling(window=TRADING_DAYS).std()*np.sqrt(TRADING_DAYS)

    fig = plt.figure(figsize=(15, 7))
    ax1 = fig.add_subplot(1, 1, 1)
    volatility.plot(ax=ax1)
    ax1.set_xlabel('Dates')
    ax1.set_ylabel('Volatility')
    ax1.set_title(f'Annualized volatility for {stock.upper()} Inc')
    plt.show()

def plot_your_volatility(df, stock):
    df["Rate of Returns"] = df["Adj Close"].pct_change()
    volatility = df["Rate of Returns"].std() * np.sqrt(252)
    volatility = round(volatility, 4)*100
    
    fig = plt.figure(figsize = (14,6))
    ax = fig.gca()
    df['Rate of Returns'].hist(ax=ax, bins=50, alpha=0.6, edgecolor='black', color='blue')
    ax.grid(False)
    ax.set_xlabel('Daily Return')
    ax.set_ylabel('Frequence')
    ax.set_title(f'{stock.upper()} volatility: {volatility}%')

def CAGR(DF):
    "function to calculate the Cumulative Annual Growth Rate of a trading strategy"
    df = DF.copy()
    df["daily_ret"] = DF["Adj Close"].pct_change()
    df["cum_return"] = (1 + df["daily_ret"]).cumprod()
    n = len(df)/252
    CAGR = (df["cum_return"][-1])**(1/n) - 1
    return CAGR

def volatility(DF):
    "function to calculate annualized volatility of a trading strategy"
    df = DF.copy()
    df["daily_ret"] = DF["Adj Close"].pct_change()
    vol = df["daily_ret"].std() * np.sqrt(252)
    return vol

def sharpe_ratio(DF):
    "function to calculate sharpe ratio ; rf is the risk free rate"
    rf = float(input('What is your current risk free rate in %: '))
    rf = rf / 100
    df = DF.copy()
    sr = (CAGR(df) - rf)/volatility(df)
    return sr

def graphique_sharpe_per_day(dataframe, nb_of_year, stock):
    if nb_of_year == 1:
        TRADING_DAYS = 252
    if nb_of_year == 3:
        TRADING_DAYS = 756
    df = dataframe.copy()
    returns = np.log(df['Close']/df['Close'].shift(1))
    returns.fillna(0, inplace=True)
    volatility = returns.rolling(window=TRADING_DAYS).std()*np.sqrt(TRADING_DAYS)
    sharpe_ratio = returns.mean()/volatility
    sharpe_ratio.tail()

    fig = plt.figure(figsize=(15, 7))
    ax3 = fig.add_subplot(1, 1, 1)
    sharpe_ratio.plot(ax=ax3)
    ax3.set_xlabel('Date')
    ax3.set_ylabel('Sharpe ratio')
    ax3.set_title(f'Sharpe ratio with the annualized volatility for {stock.upper()} Inc')
    plt.show()

def sharpe_ratio_over_10y(dataframe, rf):
    df_series = dataframe['Adj Close'].resample('Y').ffill()
    df = pd.DataFrame(df_series)

    returns = df["Adj Close"].pct_change()
    volatility = returns.rolling(window=2).std()*np.sqrt(252) # We choose a default window of 30 for the number of day in a month
    sharpe_ratio = returns - rf / volatility

    df = pd.DataFrame(sharpe_ratio)
    df = df.rename(columns={"Adj Close":"Sharpe Ratio"})
    df.reset_index(inplace=True)

    plt.figure(figsize=(16,8))
    plt.plot(df['Date'], df['Sharpe Ratio'])
    plt.xlabel('Years')
    plt.ylabel('Sharpe Ratio')
    plt.title(f'Sharpe Ratio over 10 years in {nb_of_year} frequences')
    plt.show()

try:
    #stock1 = st.text_input("Enter your first stock ticker: ")
    #stock2 = st.text_input("Enter your second stock ticker: ")
    stock1 = 'aapl'
    stock2 = 'amzn'

    # nb_of_year = st.number_input('How many year(s) span (1Y to 15Y): ', min_value=1, max_value=15, step=1)
    list_of_years = [1, 3, 5, 10]
    dic_annualized_return = {}


    for i in list_of_years:
        start = datetime.datetime.now() - datetime.timedelta(days=i*365)
        end = date.today()

        dataframe1 = web.DataReader(f'{stock1}', "yahoo", start, end)
        dataframe2 = web.DataReader(f'{stock2}', "yahoo", start, end)

    # if st.button('Accept and Launch'):
        df1 = annualized_return(dataframe1, i, stock1)
        df2 = annualized_return(dataframe2, i, stock2)

        concatcolumns_df = pd.concat([df1, df2], axis = 1)
  
        dic_annualized_return[i] = [concatcolumns_df.iloc[0][0], concatcolumns_df.iloc[0][1]]
        df_annualized_return = pd.DataFrame(dic_annualized_return)
        df_annualized_return = df_annualized_return.transpose()
        df_annualized_return = df_annualized_return.reset_index(drop=True)
        df_annualized_return.columns = [stock1, stock2]

    df_annualized_return['Years span'] = list_of_years
    st.dataframe(df_annualized_return)

except Exception as e: 
    st.write(e)
