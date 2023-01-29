import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
pd.options.mode.chained_assignment = None
tf.random.set_seed(0)


def create_train_test_windows(y,n_lookback,n_forecast):
    
    """
    Parameters:
    ------------
    
    df: Pandas df
        Stock dataframe for stock
    n_lookback: length of input sequences (lookback period)
    n_forecast: length of output sequences (forecast period)
    
    Returns:
    X,Y : train and test data in windows as X and Y 
    """
    #y = df['Close'].fillna(method='ffill')
    #y = y.values.reshape(-1, 1)
    #
    ## scale the data
    #scaler = MinMaxScaler(feature_range=(0, 1))
    #scaler = scaler.fit(y)
    #y = scaler.transform(y)
    #
    X = []
    Y = []

    for i in range(n_lookback, len(y) - n_forecast + 1):
        X.append(y[i - n_lookback: i])
        Y.append(y[i: i + n_forecast])
    
    X = np.array(X)
    Y = np.array(Y)
    
    return X, Y

def create_model(n_lookback,n_forecast):
    
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(n_lookback, 1)))
    model.add(LSTM(units=50))
    model.add(Dense(n_forecast))
    
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    return model


def make_future_dataframe(df,y,n_lookback,n_forecast,model,scaler):
    
    # generate the forecasts
    X_ = y[- n_lookback:]  # last available input sequence
    #scaler = MinMaxScaler()
    X_ = X_.reshape(1, n_lookback, 1)
    
    
    
    Y_ = model.predict(X_).reshape(-1, 1)
    Y_ = scaler.inverse_transform(Y_)
    
    # organize the results in a data frame
    df_past = df[['Close']].reset_index()
    df_past.rename(columns={'index': 'Date', 'Close': 'Actual'}, inplace=True)
    df_past['Date'] = pd.to_datetime(df_past['Date'])
    df_past['Forecast'] = np.nan
    df_past['Forecast'].iloc[-1] = df_past['Actual'].iloc[-1]
    
    df_future = pd.DataFrame(columns=['Date', 'Actual', 'Forecast'])
    df_future['Date'] = pd.date_range(start=df_past['Date'].iloc[-1] + pd.Timedelta(days=1), periods=n_forecast)
    df_future['Forecast'] = Y_.flatten()
    df_future['Actual'] = np.nan
    
    #df_future = df_future[df_future.Date.dt.weekday < 5]
    
    return df_past