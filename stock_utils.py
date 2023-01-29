import numpy as np
import pandas as pd
import yfinance as yf
import tensorflow as tf
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
pd.options.mode.chained_assignment = None
tf.random.set_seed(0)
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dropout
tf.random.set_seed(0)

def create_train_test_windows(x,y,n_lookback,n_forecast):
    
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
        X.append(x[i - n_lookback: i])
        Y.append(y[i: i + n_forecast])
    
    X = np.array(X)
    Y = np.array(Y)
    
    return X, Y

def create_model(n_lookback,n_forecast,f_len,model_type):
    
    if model_type=='LSTM':
        model = Sequential()
        model.add(LSTM(units=100, return_sequences=True, input_shape=(n_lookback, f_len)))
        model.add(LSTM(units=50))
        #model.add(Dense(50))
        model.add(Dense(25))
        model.add(Dense(n_forecast))
        
    elif model_type=='BiLSTM':
        model=Sequential()
        model.add(Bidirectional(LSTM(100, return_sequences=True, input_shape= (n_lookback, f_len))))
        model.add(Bidirectional(LSTM(50, return_sequences=False)))
        model.add(Dense(25))
        model.add(Dense(n_forecast))
            
    model.compile(loss='mean_squared_error', optimizer='adam')
    
    return model


def make_future_dataframe(df,x,y,n_lookback,n_forecast,model,scaler,features,f_len):
    
    #df.reset_index(inplace=True)
    # generate the forecasts
    X_ = x[- n_lookback:]  # last available input sequence
    #scaler = MinMaxScaler()
    X_ = X_.reshape(1, n_lookback, f_len)
    
    
    
    Y_ = model.predict(X_).reshape(-1, 1)
    Y_ = scaler.inverse_transform(Y_)
    
    
    # organize the results in a data frame
    df_past = df[['Date']+features].tail(n_lookback)
    df_past.rename(columns={'Close': 'Actual'}, inplace=True)
    #df_past.rename(columns={'index': 'Date', 'Close': 'Actual'}, inplace=True)
    df_past['Date'] = pd.to_datetime(df_past['Date'])
    df_past['Forecast'] = np.nan
    df_past['Forecast'].iloc[-1] = df_past['Actual'].iloc[-1]
    
    df_future = pd.DataFrame(columns=['Date', 'Actual', 'Forecast'])
    df_future['Date'] = pd.date_range(start=df_past['Date'].iloc[-1] + pd.Timedelta(days=1), periods=n_forecast)
    df_future['Forecast'] = Y_.flatten()
    #df_future['Actual'] = np.nan
    
    #df_future = df_future[df_future.Date.dt.weekday < 5]
    
    return df_future,df_past

def RSI(series, period):
    
    delta = series.diff().dropna()
    u = delta * 0
    d = u.copy()
    u[delta > 0] = delta[delta > 0]
    d[delta < 0] = -delta[delta < 0]
    u[u.index[period-1]] = np.mean( u[:period] ) #first value is sum of avg gains
    u = u.drop(u.index[:(period-1)])
    d[d.index[period-1]] = np.mean( d[:period] ) #first value is sum of avg losses
    d = d.drop(d.index[:(period-1)])
    rs = pd.DataFrame.ewm(u, com=period-1, adjust=False).mean() / \
         pd.DataFrame.ewm(d, com=period-1, adjust=False).mean()
    
    return 100 - 100 / (1 + rs)


def rsi_class(x):
    ret = "low"
    if x < 50:
        ret = "low"
    if x > 50:
        ret = "med"
    if x > 70:
        ret = "high"
    return(ret)

def feature_creation(df):
    """
    
    """
    df['50D-SMA'] = df['Close'].rolling(window=50).mean()    
    df['50D-EMA'] = df['Close'].ewm(span=50,adjust=False).mean()
    df['rsi'] = RSI( df['Close'], 14 )
    df['rsicat'] = list(map(rsi_class, df['rsi']))
    df.rsicat = pd.Categorical(df.rsicat)
    df['rsicat'] = df.rsicat.cat.codes
    
    df['average'] = (df['High'] + df['Low'] + df['Close'])/3
    df['vwap'] = (df['Close'] * df['Volume'])/ df['Volume']
    df['vwap_pct_ret'] = df['vwap'].pct_change()
    df['pvwap'] = df['vwap_pct_ret'].shift(-1)
    
    return df
    
def create_backtest_frames(df,n_lookback,n_forecast):
    df_leftout = df.tail(n_forecast)
    df = df.iloc[:-n_forecast]
    #df.drop(df.tail(n_forecast).index,
     #   inplace = True)
    
    return df,df_leftout