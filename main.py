#app.py
import top_mentions
import historical
import tweet_search
import predictions
import streamlit as st
PAGES = {
    "Most Mentioned Stocks": top_mentions,
    "Twitter Ticker Search":tweet_search,
    "Historical GME Analysis": historical,
    'Predictions':predictions,
}
st.sidebar.title('FinTwitter')
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
page = PAGES[selection]
page.app()
