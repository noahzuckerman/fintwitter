import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests
from sklearn.preprocessing import StandardScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow import keras
from pickle import dump,load
from datetime import datetime

def app():
    st.title('Neural Network Modelling and Predictions')

    #Bring in Datasets
    top_stocks_display= pd.read_csv('data/top_stocks_display.csv')
    top_stocks_display.set_index('stocks',inplace=True)

    df = pd.read_csv('data/df.csv')
    df.set_index('date',inplace=True)
    df.sort_index(inplace=True)
    df.index = pd.to_datetime(df.index)

    #Define functions
    def classify(word,stock):
        if word == stock:
            return 1
        else:
            return 0

    #Filter for stocks that check the flags
    stocks_to_model = top_stocks_display.loc[(top_stocks_display['outlier_flag']>0)&(top_stocks_display['delta_flag']==1)].index

    #Initialize an empty list
    stonks = []

    #If there is a candidate pullout the name onto a list
    if len(stocks_to_model)>0:
        for stocks in stocks_to_model:
            stonks.append(stocks)

    #Create dataframe including stock to model
    stock_to_predict = df.loc[df['stocks'] == stonks[0]]
    stock_to_predict['mentions'] = stock_to_predict['stocks'].apply(classify,args=(f'${stonks[0]}',)) + stock_to_predict['processed_tweets'].str.contains(f'{stonks[0]}').astype(int)
    stock_to_predict['mentions'] = stock_to_predict['mentions'].apply(lambda x: x/x if x>0 else x)
    stock_to_predict.sort_index(inplace=True)
    stock_to_predict.index = pd.to_datetime(stock_to_predict.index)

    st.write("""
    #### The following predictions we're generated on March 8th, using the last 4 weeks of twitter data. The baseline model accuracy of simply taking the average was `58%` accurate. The RNN Model displayed an 84% train accuracy and a `62%` test accuracy.
    """)

    st.write('')
    st.write("""
    #### Below are the predictions/reccomendations to buy or sell GME over the next 3 weeks. The model predicts `Buy` when it calcualtes a higher probability the price will increase 5 days from now when compared with today. If the model determines that is a low likelihood outcome, it will predict `Sell`. You can also see the realtime GME stock price displayed below.
    """)

    #Filter for a specfici time rnage of the last 1 month
    data = stock_to_predict['2021-02-08':'2021-03-08'].resample('D').sum()[['mentions','followers']]

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

    #Pre-process the twitter and price Datasets
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

    #Load pre-trained model and StandardScaler
    model = keras.models.load_model('assets/rrn_model.csv')
    scaler = load(open('assets/scaler.pkl','rb'))

    #Create X and Y
    X = merge.drop(columns=['target','mentions'])
    y = merge['target'].values

    #Scale the data
    X_sc = scaler.transform(X)

    #Convert data to a series using Time Series Generator
    test_sequences = TimeseriesGenerator(X_sc,y,length=3,batch_size=128)

    #Make Predictions
    preds = model.predict_classes(test_sequences)

    #Create dates for prediction dataframe - starting at last date of
    #avaialble data
    dates = pd.date_range(start='2021-03-09',end='2021-04-03')
    predictions = dates.to_frame()
    predictions.index = predictions.index.date

    #Add predictions into dataframe
    predictions['preds'] = preds
    predictions.drop([0],axis=1,inplace=True)
    predictions['preds'] = predictions['preds'].map({0:'Sell',1:'Buy'})

    #create gird
    col1, col2 = st.beta_columns((0.5,1.5))

    #Display the dataframe of predictions
    col1.write('')
    col1.write(predictions)

    col2.write('')

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
    df['price'] =df['4. close']
    price_chart= df['price']
    idx = pd.date_range('2021-02-08', '2021-03-10')
    price_chart = price_chart.reindex(idx,method='nearest')

    #Plot the price on a chart
    col2.line_chart(price_chart)
