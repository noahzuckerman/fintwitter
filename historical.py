import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

def app():
    st.title('Historical GME Analysis')
    st.write("""
    #### In order to get an understanding of the scale of the GME outlier event, I analyzed the last 3 months of historical twitter data and over 500K tweets to infer some statistical measures to identify another black swan event.
    """)

    #Bring in datasets
    tweets= pd.read_csv('data/twitter_historical_05.03.21_pre-processed.csv')

    data= pd.read_csv('data/twitter_historical_05.03.21_pre-processed_agg_time.csv')
    data.set_index('date',inplace=True)
    data.sort_index(inplace=True)
    data.index = pd.to_datetime(data.index)

    price= pd.read_csv('data/gme_price.csv')
    price.rename(columns = {'Unnamed: 0':'date'},inplace=True)
    price.set_index('date',inplace=True)
    price.sort_index(inplace=True)
    price.index = pd.to_datetime(price.index)

    merge1 = data[['mentions','followers']]
    merge2 = price
    merge=pd.merge(merge1,merge2, how='inner', left_index=True, right_index=True)
    merge.rename(columns={
        '4. close':'price'
    },inplace=True)

    corr = round(merge.corr()['mentions']['price'],2)
    correlation = str(corr)[2] + str(corr)[3] + '%'

    st.write('')

    with st.beta_expander("See historical GME mentions vs. price"):
        #create gird
        col1, col2, col3 = st.beta_columns((0.7,0.3,1.4))

        #Print correlation
        col2.markdown(f"<h1 style='text-align: left; color: cornflowerblue;font-size:36px;'>{correlation}</h1>", unsafe_allow_html=True)
        col3.write(' ')
        col3.write(' ')
        col3.write('Linear Correlation')

        #Plot time series chart
        fig,ax1 = plt.subplots(figsize=(16,11))
        right_side = ax1.spines["right"]
        right_side.set_visible(False)
        upper_side = ax1.spines["top"]
        upper_side.set_visible(False)

        ax2 =ax1.twinx()
        upper_side = ax2.spines["top"]
        upper_side.set_visible(False)

        #Axis 1
        color = 'dimgray'
        #ax1.set_ylabel('Observed GME mentions',color=color)
        l1 = ax1.plot(data['mentions'],color=color,label='Observed GME mentions')
        ax1.tick_params(axis='y',labelcolor=color,labelsize=16)
        ax1.tick_params(axis='x',labelsize=14)
        #ax1.legend(loc=2,fontsize=16)

        #Axis 2
        color = 'cornflowerblue'
        #ax2.set_ylabel('GME Closing Price ($)',color=color)
        l2 = ax2.plot(price,color=color,label='GME Closing Price ($)')
        ax2.tick_params(axis='y',labelcolor=color,labelsize=16)
        fig.legend([l1,l2],labels=['Observed GME mentions','GME Closing Price ($)'],loc='lower left',bbox_to_anchor=(0.56,0.75),fontsize=16)
        #ax2.legend(loc=1,fontsize=16)
        st.pyplot(fig);

    st.write('')

    data['sev_day_ma'] = data['mentions'].rolling(7).mean()
    merge4 = data['sev_day_ma']
    merged_ma=pd.merge(merge2,merge4, how='inner', left_index=True, right_index=True)
    merged_ma.rename(columns={
        '4. close':'price'
    },inplace=True)

    corr = round(merged_ma.corr()['sev_day_ma']['price'],2)
    correlation = str(corr)[2] + str(corr)[3] + '%'

    st.write('')

    with st.beta_expander("Seven day moving average of GME mentions vs. price"):
        #create gird
        col1, col2, col3 = st.beta_columns((0.7,0.3,1.4))

        #Print correlation
        col2.markdown(f"<h1 style='text-align: left; color: cornflowerblue;font-size:36px;'>{correlation}</h1>", unsafe_allow_html=True)
        col3.write(' ')
        col3.write(' ')
        col3.write('Linear Correlation')

        #Plot time series chart
        fig,ax1 = plt.subplots(figsize=(16,11))
        right_side = ax1.spines["right"]
        right_side.set_visible(False)
        upper_side = ax1.spines["top"]
        upper_side.set_visible(False)

        ax2 =ax1.twinx()
        upper_side = ax2.spines["top"]
        upper_side.set_visible(False)

        #Axis 1
        color = 'dimgray'
        #ax1.set_ylabel('Seven day moving average',color=color)
        l1 = ax1.plot(data['sev_day_ma'],color=color,label = 'Seven day moving average')
        ax1.tick_params(axis='y',labelcolor=color,labelsize=16)
        ax1.tick_params(axis='x',labelsize=14)
        #ax1.legend(loc=2,fontsize=16)

        #Axis 2
        color = 'cornflowerblue'
        #ax2.set_ylabel('GME Closing Price ($)',color=color)
        l2 = ax2.plot(price,color=color,label='GME Closing Price ($)')
        ax2.tick_params(axis='y',labelcolor=color,labelsize=16)
        #ax2.legend(loc=1,fontsize=16)
        fig.legend([l1,l2],labels=['Seven day moving average','GME Closing Price ($)'],loc='lower left',bbox_to_anchor=(0.56,0.75),fontsize=16)
        st.pyplot(fig);
