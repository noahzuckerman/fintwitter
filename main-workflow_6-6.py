#!/usr/bin/env python
# coding: utf-8

# In[36]:


#Load packages from tweepy
import tweepy as tp

#Load package from snscrape to scrape twitters frontend
import snscrape.modules.twitter as sntwitter

#NLP Modules
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#Streamlit
import streamlit as st

#Genism module
import gensim

#Import matplot
import matplotlib.pyplot as plt
import seaborn as sns

#Load additional packages
import time
import datetime
import numpy as np
import pandas as pd
import requests
from collections import Counter
import itertools
import re
import itertools
import sys
from retry import retry
import os
import dotenv

#Statsmodels
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf

pd.set_option('display.max_colwidth', None)


# ### Define functions

# Function to classify and count mentions of a given stock

# In[5]:


def classify(word,stock):
    if word == stock:
        return 1
    else:
        return 0


# Function to convert timestamp to a datetime object

# In[6]:


def to_time(t):
    date_time_obj = datetime.datetime.strptime(t,'%Y-%m-%d %H:%M:%S').date()
    return date_time_obj


# ### Read in pre-processed historical data

# Read in granular tweet level data

# In[7]:


df_gran = pd.read_csv('data/twitter_historical_05.03.21_pre-processed.csv')
df_gran.dropna(subset = ['processed_tweets'],inplace=True)
df_gran.drop(['Unnamed: 0'],axis=1,inplace=True)
df_gran.drop(['weighted_mentions'],axis=1,inplace=True)
df_gran.drop(['mentions'],axis=1,inplace=True)
df_gran.drop(['id'],axis=1,inplace=True)
df_gran.set_index('date',inplace=True)
df_gran.sort_index(inplace=True)
df_gran.dropna(inplace=True)
df_gran.index = pd.to_datetime(df_gran.index)
df_gran['retweets'] = np.nan
df_gran['like'] = np.nan


# In[8]:


#Read in latest live twitter data
df_live = pd.read_csv('data/twitter_live_2021-03-08.csv')
df_live.dropna(subset = ['processed_tweets'],inplace=True)
df_live.drop(['Unnamed: 0'],axis=1,inplace=True)


# In[9]:


#Instantiate sentiment analyzer 
sid = SentimentIntensityAnalyzer()


# In[10]:


#Transform sentiment scores, date and processed tweets columns
df_live['scores'] = df_live['tweet'].apply(lambda tweet: sid.polarity_scores(tweet))
df_live['compound']  = df_live['scores'].apply(lambda score_dict: score_dict['compound'])
df_live['sentiment'] = df_live['compound'].apply(lambda c: 1 if c >0 else (0 if c==0 else -1))
df_live['created_at_transformed'] = df_live['created_at'].map(lambda x: x[:-6] if len(x)>19 else x)
df_live['date'] = df_live['created_at_transformed'].apply(to_time)
df_live.dropna(subset = ['processed_tweets'],inplace=True)
df_live['processed_tweets'] = df_live['processed_tweets'].apply(lambda x: x.lower() if type(x) ==str else x)
df_live.set_index('date',inplace=True)
df_live.sort_index(inplace=True)
df_live.dropna(inplace=True)
df_live.index = pd.to_datetime(df_live.index)


# In[11]:


#Combine historical and latest twitter data
df = df_gran.append(df_live)


# ## Match stock names to twitter matches

# In[12]:


#Import list of stock symbols
df_syms = pd.read_csv('data/stock_names.csv')
df_syms=df_syms['Symbol'].str.lower()

sym_list = []
for x in df_syms:
    sym_list.append(x)
    
df['stocks'] = df['stocks'].apply(lambda sym: sym if sym in sym_list else np.nan)
df_stocks = df.copy()
df_stocks.dropna(subset=['stocks'],inplace=True)
df_stocks.sort_index(inplace=True)


# ### Create a rolling 30 day dataframe for latest snapshot data

# In[13]:


#Filter entire list for the last week
df_stocks_recent = df_stocks.loc['2021-03-01':'2021-03-08']

#Find the stocks with the 15 most mentions in the last week
top_stocks = df_stocks_recent.groupby('stocks').count().sort_values(by='tweet',ascending=False)['screen_name'].head(15)
top_stocks = pd.DataFrame(top_stocks)

#Group the top stocks 
stocks = df_stocks_recent.groupby('stocks').agg(['sum','count'])[['followers','compound','sentiment']]
stocks['overall_sentiment'] = stocks['compound']['sum']/stocks['compound']['count']
top_stocks_display=pd.merge(top_stocks,stocks, how='inner', left_index=True, right_index=True)
top_stocks_display = top_stocks_display[['screen_name',top_stocks_display.columns[1],top_stocks_display.columns[7]]]
top_stocks_display.columns=['mentions','audience','sentiment']
top_stocks_display  = top_stocks_display.head(10)
top_stocks_list = list(top_stocks_display.index)


# In[14]:


top_stocks_display


# ### Calculate overall stock mention sentiment

# In[15]:


market_sentiment = round(df_stocks['sentiment'].sum()/df_stocks.shape[0],2)
market_sent = str(market_sentiment)[2] + str(market_sentiment)[3] + '%'
market_sent


# ### Top 10 stocks as time series

# In[16]:


for stock in top_stocks_list:
    df_stocks[f'{stock}'] = df_stocks['stocks'].apply(classify,args=(f'${stock}',)) + df_stocks['processed_tweets'].str.contains(f'{stock}').astype(int)
    df_stocks[f'{stock}'] = df_stocks[f'{stock}'].apply(lambda x: x/x if x>0 else x)


# In[17]:


plt.figure(figsize=(16,9))

for stock in top_stocks_list:
    plt.plot(df_stocks['2021'].resample('D').sum()[stock],label=stock)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel('Observed mentions of the stock symbol in tweets',fontsize=14)
    plt.legend(loc="upper right",fontsize=12);


# ### Calculate GME flagging logic

# Statsitical outlier flagging

# In[18]:


fig,axes = plt.subplots(nrows = len(top_stocks_list),
                            figsize=(8,6*len(top_stocks_list))) #Vertical image size needs to scale dunamically based on 
                                                                #number of dpts
for i,stock in enumerate(top_stocks_list):
    sns.boxplot(ax=axes[i],data = df_stocks['2021'].resample('D').sum()[stock])
    axes[i].set_title(f'{stock}');


# In[19]:


#Filter dataframe
outliers = df_stocks['2021'].resample('D').sum()[top_stocks_list]
#Calculate outliers
Q1=outliers.quantile(0.25)
Q3=outliers.quantile(0.75)
IQR=Q3-Q1
outlier_count = round((((outliers < (Q1 - 1.5 * IQR)) | (outliers > (Q3 + 1.5 * IQR))).sum()/outliers.count()).sort_values(ascending=False),2)
outlier_count


# In[20]:


out = outlier_count[outlier_count>0.09]
names = out.index
outlier_dic = {k:v for k,v in zip(names,out)}
top_stocks_display['outlier_flag'] = top_stocks_display.index.map(outlier_dic)


# In[21]:


top_stocks_display


# GME 7 day rolling avg flag

# In[22]:


#Assing a flag based on the threshold set by GME's 7 day rolling average
delta_flag_list = []
for stock in top_stocks_list:
    outliers[f'{stock}_sev_day_ma'] = outliers[stock].rolling(7).mean()
    outliers[f'{stock}_delta_flag'] = outliers[f'{stock}_sev_day_ma'].apply(lambda x: 1 if x>25 else 0)
    delta_flag_list.append(f'{stock}_delta_flag')
    
#Summary dataframe of statistics to identify any columns with greater than 0 in the flag column
summary = outliers.describe()[delta_flag_list]


#Assign the flag to to the top stocks in a dictionary
delta_flag_dic = {}
for col in summary:
    flag = summary[col]['max']
    string = col.replace('_delta_flag','')
    delta_flag_dic[string] = flag

#Map dictionary to dataframe
top_stocks_display['delta_flag'] = top_stocks_display.index.map(delta_flag_dic)


# In[23]:


top_stocks_display


# ### User defined contains search to filter tweets

# In[24]:


def search_tweets(string,df):
    df = df[['screen_name','tweet','followers','retweets','processed_tweets','like']]
    tweet_display = df.loc[df['processed_tweets'].str.contains(f'{string}')].sort_values('followers',ascending=False)[['screen_name','followers','tweet']]
    tweet_display.sort_index(inplace=True,ascending=False)
    return tweet_display.head(5)


# In[25]:


search_tweets('gamestop',df_stocks_recent)


# ### Modelling/ Predictions

# Data prep for modelling

# In[26]:


stocks_to_model = top_stocks_display.loc[(top_stocks_display['outlier_flag']>0)&(top_stocks_display['delta_flag']==1)].index


# In[27]:


stonks = []
if len(stocks_to_model)>0:
    for stocks in stocks_to_model:
        stonks.append(stocks)
        
stock_to_predict = df.loc[df['stocks'] == stonks[0]]
stock_to_predict['mentions'] = stock_to_predict['stocks'].apply(classify,args=(f'${stonks[0]}',)) + stock_to_predict['processed_tweets'].str.contains(f'{stonks[0]}').astype(int)
stock_to_predict['mentions'] = stock_to_predict['mentions'].apply(lambda x: x/x if x>0 else x)
stock_to_predict.sort_index(inplace=True)
stock_to_predict.index = pd.to_datetime(stock_to_predict.index)


# In[28]:


data = stock_to_predict['2021-02-08':'2021-03-08'].resample('D').sum()[['mentions','followers']]


# Pull stock price data

# In[29]:


#Pull price data from alphavantage api
base_url = "https://www.alphavantage.co/query"
req_av = requests.get(base_url,params={'function':'TIME_SERIES_DAILY',
                                       'symbol':f'{stonks[0]}',
                                      'apikey':'QMUU6CQUKPM70QZ5','outputsize':'full'}) 

req_av.status_code
req_av.url
gme = req_av.json()
gme['Time Series (Daily)']
df = pd.DataFrame(gme['Time Series (Daily)'])
df = df.T
df.index = pd.to_datetime(df.index)
df = df.sort_index()
df = df['2021']
df['1. open'] = pd.to_numeric(df['1. open'], downcast="float")
df['2. high'] = pd.to_numeric(df['2. high'], downcast="float")
df['3. low'] = pd.to_numeric(df['3. low'], downcast="float")
df['4. close'] = pd.to_numeric(df['4. close'], downcast="float")
df['5. volume'] = pd.to_numeric(df['5. volume'], downcast="float")
s= df['4. close']
idx = pd.date_range('2021-02-08', '2021-03-08')
s = s.reindex(idx,method='nearest')


# In[30]:


#Merge data and price data
merge=pd.merge(data,s, how='inner', left_index=True, right_index=True)
merge.rename(columns={
    '4. close':'price'
},inplace=True)

#Add target column for model predictions
merge['target'] = merge['price'].diff(5).apply(lambda r: 1 if r >0 else 0)

#Remove any trend fromt the price data
merge['price'] = merge['price'].pct_change()

#Create a cumulative sum
merge['cum_mentions'] = merge['mentions'].cumsum()


# In[31]:


#Calculate baseline score
baseline = merge['target'].mean()
baseline


# In[32]:


#Define X and y
X = merge.drop(columns=['target','mentions'])
y = merge['target'].values

#Generate a time series sequence
test_sequences = TimeseriesGenerator(X,y,length=3,batch_size=128)

#Create predictions
model.predict_classes(test_sequences)


# ### Streamlit Frontend

# In[35]:


st.title('Twitter Buzz Tool')


# In[ ]:




