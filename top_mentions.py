import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def app():
    st.title('Most Mentioned Stocks')

    #Bring in Datasets
    top_stocks_display= pd.read_csv('data/top_stocks_display.csv')
    top_stocks_display.set_index('stocks',inplace=True)
    top_stocks_list = list(top_stocks_display.index)
    top_stocks_list.insert(0,'All')

    df_stocks = pd.read_csv('data/df_stocks')
    df_stocks.set_index('date',inplace=True)
    df_stocks.sort_index(inplace=True)
    df_stocks.index = pd.to_datetime(df_stocks.index)

    st.write("""
    #### The following stocks are the top 10 most mentioned across the searched twitter accounts and second layer network searches. This is measured over the last week of data.
    """)
    #Sdeibar title/seperator
    st.sidebar.write('')
    st.sidebar.title('Ticker Display Filters')

    #Type in any string to search
    user_input = st.sidebar.multiselect("Select companies to display on chart",top_stocks_list,default='All')
    st.write('')

    st.dataframe(top_stocks_display)

    st.write('')

    if 'All' in user_input:
        top_stocks_list.remove('All')
    else:
        top_stocks_list = user_input

    def plot_mentions(top_stocks_list):

        fig, ax = plt.subplots(figsize=(18,12))
        right_side = ax.spines["right"]
        right_side.set_visible(False)

        upper_side = ax.spines["top"]
        upper_side.set_visible(False)

        for stock in top_stocks_list:
            plt.plot(df_stocks.loc['2021'].resample('D').sum()[stock],label=stock)
            plt.xticks(fontsize=18)
            plt.yticks(fontsize=18)
            plt.legend(loc="upper right",fontsize=18);
        return st.pyplot(fig)

    st.markdown("<h1 style='text-align: center; color: black;font-size:16px;'>Observed frequency of mentions over time</h1>", unsafe_allow_html=True)
    st.write('')
    plot_mentions(top_stocks_list)
