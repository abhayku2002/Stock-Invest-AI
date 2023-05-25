import streamlit as st
from datetime import date
import matplotlib.pyplot as plt
import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objs as go
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

START = "2015-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

st.title('Smart Invest AI') 

st.header('Stock prediction System with Machine Learning using Streamlit ')
st.write('This data is collected from yahoo finance and the prediction is based on the past data of the stock')


stocks = ('UNH', 'JNJ', 'LLY', 'MRK','ABT','ELV','ZTS','SYK','CVS','GILD','MDT')
selected_stock = st.selectbox('Select dataset for  Healthcare prediction', stocks)

n_years = st.slider('Years of prediction:', 1, 5)
period = n_years * 365


@st.cache_data
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data

	
data_load_state = st.text('Loading data...')
data = load_data(selected_stock)
data_load_state.text('Loading data... done!')

st.subheader('Raw data : First 5 entries')
st.write(data.head())

st.subheader('Raw data : Last 5 entries')
st.write(data.tail())

st.subheader('Dataset Infomation')
st.write(data.describe())


def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name="stock_open",line_color='red'))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close",line_color='green'))
	fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)
	
plot_raw_data()

df_train = data[['Date','Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

X = data[['Open', 'High', 'Low', 'Close', 'Volume']]
y = data['Adj Close']

model = LinearRegression()
model.fit(X, y)

# Predict the stock price
prediction = model.predict([[100, 200, 300, 400, 500]])

# Display the prediction
st.write('**The predicted stock price is:**', prediction)


m = Prophet()
m.fit(df_train)
future = m.make_future_dataframe(periods=period)
forecast = m.predict(future)


st.subheader('Forecast data : First 5 Entries ')
st.write(forecast.head())

st.subheader('Forecast data : last 5 Entries ')
st.write(forecast.tail())
    
st.write(f'Forecast plot for {n_years} years')
fig1 = plot_plotly(m, forecast)
st.plotly_chart(fig1)
#---09st.plotly_chart(fig2)


st.write("Forecast components")
fig2 = m.plot_components(forecast)
st.write(fig2)

