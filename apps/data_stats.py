import streamlit as st
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import pandas_datareader as web
import datetime
import base64
from datetime import date
from multiapp import MultiApp
import streamlit.components.v1 as components


#------------------------------------------------------------------Functions----------------------------------------------------------------------------------------

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

    ax1.plot(df1['Normed Return'], c='blue', label=stock1)
    ax2.plot(df2['Normed Return'], c='green', label=stock2)
    
    ax1.set_xlabel('Years')
    ax2.set_xlabel('Years')
    
    ax1.set_ylabel('Cumulatize Return')
    ax2.set_ylabel('Cumulatize Return')
    
    ax1.set_title(f'{stock1.upper()} price over 10 years')
    ax2.set_title(f'{stock2.upper()} price over 10 years')

    ax1.legend(loc='upper left')
    ax2.legend(loc='upper left')
    st.pyplot(fig)



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

    plt.title(f'{stock1.upper()} vs {stock2.upper()} price over 10 years')
    plt.xlabel('Years')
    plt.ylabel('Cumulative Return')

    plt.legend(loc='upper left')
    st.pyplot(fig)
    
    return df1, df2


def rendement_mensuel(stock, nb_of_year):
    # CODE À VALIDER (POUR LE RESAMPLE, QUI EST MONTHLY)

    start = datetime.datetime.now() - datetime.timedelta(days=10*365)
    end = date.today()
    dataframe = web.DataReader(f'{stock}', "yahoo", start, end)

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
    
    fig = plt.figure(figsize=(10,5))
    plt.plot(df['Date'], df[f'Annualized Return on {nb_of_year} years'], color='purple')
    plt.xlabel('Years (Month Frequency)')
    plt.ylabel('Annualized Monthly Return')
    plt.title(f'{stock.upper()} since 10 years ago')
    st.pyplot(fig)

    return df



def rendement_annualiser(stock, nb_of_year):

    start = datetime.datetime.now() - datetime.timedelta(days=10*365)
    end = date.today()
    dataframe = web.DataReader(f'{stock}', "yahoo", start, end)

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
    
    current_date = date.today()
    df['Date'].loc[10] = current_date
    df['Date'] = pd.to_datetime(df['Date'], format="%Y-%m-%d")
    
    fig = plt.figure(figsize=(20,10))
    plt.plot(df['Date'], df[f'Annualized Return on {nb_of_year} years (%)'], color='green')
    plt.xlabel('Years')
    plt.ylabel(f'Annualized Return on {nb_of_year} years')
    plt.title(f'Annualized Return of {stock.upper()} since 10 years ago')
    st.pyplot(fig)
      
    return df 



def volatility_for_10y(stock):

    start = datetime.datetime.now() - datetime.timedelta(days=10*365)
    end = date.today()
    dataframe = web.DataReader(f'{stock}', "yahoo", start, end)

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
    st.pyplot(fig)

    return volatility



def hist_volatility(stock):

    start = datetime.datetime.now() - datetime.timedelta(days=10*365)
    end = date.today()
    df = web.DataReader(f'{stock}', "yahoo", start, end)

    df["Rate of Returns"] = df["Adj Close"].pct_change()
    volatility = df["Rate of Returns"].std() * np.sqrt(252*10)  # TO CONFIRM: Do we need to put a 10 years multiplicator on the square root because it's a 10 years timeframe?
    volatility = round(volatility, 4)*100
    
    fig = plt.figure(figsize = (14,6))
    ax = fig.gca()
    df['Rate of Returns'].hist(ax=ax, bins=50, alpha=0.6, edgecolor='black', color='blue')
    ax.grid(False)
    ax.set_xlabel('Daily Return')
    ax.set_ylabel('Frequence')
    ax.set_title(f'{stock.upper()} volatility: {volatility}%')
    st.pyplot(fig)



def graphique_sharpe_per_day(stock, nb_of_year):

    start = datetime.datetime.now() - datetime.timedelta(days=10*365)
    end = date.today()
    dataframe = web.DataReader(f'{stock}', "yahoo", start, end)

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
    ax3.set_title(f'Daily sharpe ratio for {stock.upper()}')
    st.pyplot(fig)

    return sharpe_ratio



def sharpe_ratio_over_10y(stock, rf):

    start = datetime.datetime.now() - datetime.timedelta(days=10*365)
    end = date.today()
    dataframe = web.DataReader(f'{stock}', "yahoo", start, end)

    df_series = dataframe['Adj Close'].resample('Y').ffill()
    df = pd.DataFrame(df_series)

    returns = df["Adj Close"].pct_change()
    volatility = returns.rolling(window=2).std()*np.sqrt(252) # We choose a default window of 30 for the number of day in a month
    sharpe_ratio = returns - rf / volatility
    rf_percent= rf * 100

    df = pd.DataFrame(sharpe_ratio)
    df = df.rename(columns={"Adj Close":"Sharpe Ratio"})
    df.reset_index(inplace=True)

    fig = plt.figure(figsize=(16,8))
    plt.plot(df['Date'], df['Sharpe Ratio'], c='green')
    plt.xlabel('Years')
    plt.ylabel('Sharpe Ratio')
    plt.title(f'Sharpe Ratio over 10 years for {stock.upper()}, (Risk Free Return: {rf_percent} %)')
    st.pyplot(fig)



def get_table_download_link_csv(df, name_file, stock_name):
    #csv = df.to_csv(index=False)
    csv = df.to_csv().encode()
    #b64 = base64.b64encode(csv.encode()).decode() 
    b64 = base64.b64encode(csv).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{name_file} for {stock_name}.csv" target="_blank">Download csv file for {stock_name.upper()}</a>'
    return href



#------------------------------------------------------------------Code Usage----------------------------------------------------------------------------------------


def app():
    try:
        
        stock1 = st.text_input("Enter your first stock ticker: ")
        stock2 = st.text_input("Enter your second stock ticker: ")
        #stock1 = 'amzn'
        #stock2 = 'aapl'
        #nb_of_year = st.number_input('How many year(s) span (1Y to 15Y): ', min_value=1, max_value=15, step=1)
        dropdown = st.selectbox('Analysis Type: ',  ['', 'Prices', 'Monthly Returns', 'Annualized Returns', 'Volatility', 'Sharpe Ratio'], format_func=lambda x: 'Select an option' if x == '' else x)

        if dropdown == 'Prices':

            df1, df2 = one_graph_plot(stock1, stock2)
            components.html("""<hr style="height:5px;border:none;color:#333;background-color:#333;" /> """)
            two_graph_plot(stock1, stock2)
            st.markdown(get_table_download_link_csv(df1, 'Prices', stock1), unsafe_allow_html=True)
            st.markdown(get_table_download_link_csv(df2, 'Prices', stock2), unsafe_allow_html=True)

        if dropdown == 'Monthly Returns':

            df_vol_1 = rendement_mensuel(stock1, 10)
            components.html("""<hr style="height:5px;border:none;color:#333;background-color:#333;" /> """)
            df_vol_2 = rendement_mensuel(stock2, 10)
            st.markdown(get_table_download_link_csv(df_vol_1, 'Monthly Returns', stock1), unsafe_allow_html=True)
            st.markdown(get_table_download_link_csv(df_vol_2, 'Monthly Returns', stock2), unsafe_allow_html=True)

        if dropdown == 'Annualized Returns':

            radio_button = st.radio('Select timeframe: ', ('1Y', '3Y', '5Y'))
            if radio_button == '1Y':
                df1_1Y = rendement_annualiser(stock1, 1)
                df2_1Y = rendement_annualiser(stock2, 1)
                st.markdown(get_table_download_link_csv(df1_1Y, 'Annualized Returns on 1Y', stock1), unsafe_allow_html=True)
                st.markdown(get_table_download_link_csv(df2_1Y, 'Annualized Returns on 1Y', stock2), unsafe_allow_html=True)
            elif radio_button == '3Y':
                df1_3Y = rendement_annualiser(stock1, 3)
                df2_3Y = rendement_annualiser(stock2, 3)
                st.markdown(get_table_download_link_csv(df1_3Y, 'Annualized Returns on 3Y', stock1), unsafe_allow_html=True)
                st.markdown(get_table_download_link_csv(df2_3Y, 'Annualized Returns on 3Y', stock2), unsafe_allow_html=True)
            else:
                df1_5Y = rendement_annualiser(stock1, 5)
                df2_5Y = rendement_annualiser(stock2, 5)
                st.markdown(get_table_download_link_csv(df1_5Y, 'Annualized Returns on 5Y', stock1), unsafe_allow_html=True)
                st.markdown(get_table_download_link_csv(df2_5Y, 'Annualized Returns on 5Y', stock2), unsafe_allow_html=True)
        
        if dropdown == 'Volatility':

            hist_volatility(stock1)
            df1_vol = volatility_for_10y(stock1)
            components.html("""<hr style="height:5px;border:none;color:#333;background-color:#333;" /> """)
            hist_volatility(stock2)
            df2_vol = volatility_for_10y(stock2)

            st.markdown(get_table_download_link_csv(df1_vol, 'Volatility', stock1), unsafe_allow_html=True)
            st.markdown(get_table_download_link_csv(df2_vol, 'Volatility', stock2), unsafe_allow_html=True)
        
        if dropdown == 'Sharpe Ratio':
            radio_button = st.radio('Select timeframe: ', ('1Y', '3Y'))
            if radio_button == '1Y':

                df1_sharpe_1y = graphique_sharpe_per_day(stock1, 1)
                sharpe_ratio_over_10y(stock1, 0.025)
                components.html("""<hr style="height:5px;border:none;color:#333;background-color:#333;" /> """)
                df2_sharpe_1y = graphique_sharpe_per_day(stock2, 1)
                sharpe_ratio_over_10y(stock2, 0.025)

                st.markdown(get_table_download_link_csv(df1_sharpe_1y, 'Sharpe Ratio on 1Y', stock1), unsafe_allow_html=True)
                st.markdown(get_table_download_link_csv(df2_sharpe_1y, 'Sharpe Ratio on 1Y', stock2), unsafe_allow_html=True)

            else:

                df1_sharpe_3y = graphique_sharpe_per_day(stock1, 3)
                sharpe_ratio_over_10y(stock1, 0.025)
                components.html("""<hr style="height:5px;border:none;color:#333;background-color:#333;" /> """)
                df2_sharpe_3y = graphique_sharpe_per_day(stock2, 3)
                sharpe_ratio_over_10y(stock2, 0.025)

                st.markdown(get_table_download_link_csv(df1_sharpe_3y, 'Sharpe Ratio on 3Y', stock1), unsafe_allow_html=True)
                st.markdown(get_table_download_link_csv(df2_sharpe_3y, 'Sharpe Ratio on 3Y', stock2), unsafe_allow_html=True)

            
    except Exception as e: 
        st.write(e)

