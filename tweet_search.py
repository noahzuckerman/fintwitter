import streamlit as st
import pandas as pd

#Define navaigation fucntion
def app():
    st.title('Twitter Ticker Search')
    #Bring in Datasets
    df_stocks_recent = pd.read_csv('data/df_stocks_recent')

    st.write("""
    #### The following tweets are pulled from twitter daily. The table below is sorted from highest followers to lowest.
    """)
    #Sdeibar title/seperator
    st.sidebar.write(' ')
    st.sidebar.title('Ticker Search Filters')

    #Type in any string to search
    user_input = st.sidebar.text_input("Search for a company using it's ticker (all lowercase)",'gme')

    #Select values from drop down
    rows_display = st.sidebar.slider("Rows to display",5,50)

    #Define functions
    def search_tweets(string,df,depth):
        df = df[['screen_name','tweet','followers','retweets','processed_tweets','like','compound','sentiment']]
        tweet_display = df.loc[df['processed_tweets'].str.contains(f'{string}')].sort_values('followers',ascending=False)[['screen_name','followers','tweet','compound','sentiment']]
        market_sentiment = round(tweet_display['sentiment'].sum()/tweet_display.shape[0],2)
        market_sent = '{0:.0%}'.format(market_sentiment)
        tweet_display['sentiment'] = tweet_display['sentiment'].map({1:'+ve',0:'Neutral',-1:'-ve'})
        tweet_display.sort_index(inplace=True,ascending=False)
        return tweet_display.head(depth),market_sent

    #Call the fucntion
    search_display,market_sent = search_tweets(user_input,df_stocks_recent,rows_display)

    #create gird
    col1, col2 = st.beta_columns((1.9,1.4))

    #Display search results
    #col2.title(f'{market_sent}')

    col1.write('')
    col1.write('')
    col2.markdown(f"<h1 style='text-align: left; color: cornflowerblue;font-size:36px;'>{market_sent}</h1>", unsafe_allow_html=True)
    col1.write(f'Compund market sentiment for {user_input} over last week:')

    with st.beta_expander("Show top relevant tweets"):
        st.table(search_display)
