import streamlit as st
from datetime import date
import numpy as np
import yfinance as yf
from fbprophet import Prophet
from fbprophet.plot import plot_plotly
from plotly import graph_objs as go
import stock_utils
from stock_utils import *
from datetime import datetime

START = "2010-01-01"
TODAY = datetime.now()


st.title('Stock Forecast App')

window_selection_c = st.sidebar.container() # create an empty container in the sidebar
window_selection_c.markdown("## Insights") # add a title to the sidebar container
#sub_columns = window_selection_c.columns(2) #Split the container into two columns for start and end date

STOCKS = np.array([ "GOOG","AMZN","AAPL"])  # TODO : include all stocks
SYMB = window_selection_c.selectbox("select stock", STOCKS)

#stocks = ('GOOG', 'AAPL', 'MSFT', 'GME')
#selected_stock = st.selectbox('Select dataset for prediction', stocks)

#chart_width = st.expander(label="chart width").slider("", 1,4)

#period = st.slider('Days of prediction:', 10, 60)
#period = n_years * 365

st.sidebar.markdown("## Forecasts")
train_test_forecast_c = st.sidebar.container()


TRAIN_INTERVAL_LENGTH = train_test_forecast_c.number_input(
    "number of  day to use for training in window",
    min_value=100,
    key="TRAIN_INTERVAL_LENGTH",
)


TEST_INTERVAL_LENGTH = train_test_forecast_c.number_input(
    "number of days to Forecast",   
    min_value=10,
    key="TEST_INTERVAL_LENGTH",
)

train_test_forecast_c.button(
    label="Train",
    key='TRAIN_JOB'
)

@st.cache
def load_data(ticker):
    data = yf.download(ticker, START, TODAY)
    data.reset_index(inplace=True)
    return data


data_load_state = st.text('Loading data...')
data = load_data(SYMB)
data_load_state.text('Loading data... done!')

st.subheader('Raw data')
st.write(data.head())

# Plot raw data
def plot_raw_data():
    fig = go.Figure()
    #fig.add_trace(go.Scatter(x=df_train['Date'], y=df_train['Open'], name="stock_open"))
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_close",connectgaps=True))
    fig.layout.update(title_text='Time Series data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)

def plot_forecast():
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name="stock_open",connectgaps=True))
    fig.add_trace(go.Scatter(x=df_f['Date'], y=df_f['Forecast'], name="stock_forecast",connectgaps=True))
    fig.layout.update(title_text='Forecast data with Rangeslider', xaxis_rangeslider_visible=True)
    st.plotly_chart(fig)
    
#plot_raw_data()

## Predict forecast with Prophet.
#df_train = data[['Date','Close']]
##df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})
plot_raw_data()
y = data['Close'].fillna(method='ffill')
y = y.values.reshape(-1, 1)


#
## generate the input and output sequences
n_lookback = TRAIN_INTERVAL_LENGTH  # length of input sequences (lookback period)
n_forecast = TEST_INTERVAL_LENGTH  # length of output sequences (forecast period)
#
if st.session_state.TRAIN_JOB:
    text=st.empty() # Because streamlit adds widgets sequentially, we have to reserve a place at the top (after the chart of part 1)
    
    bar=st.empty() # Reserve a place for a progess bar
    
    text.write('Training model ... ') 
    bar=st.progress(0)
    
    data = feature_creation(data)
    data.bfill(inplace=True)
    
    # scale the data
    
    x = data[['Open','High','Low','Close','50D-SMA', '50D-EMA', 'rsi']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    x = scaler.fit_transform(x)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(y)
    y = scaler.transform(y)
    
    
    #X = []
    #Y = []
    #
    #for i in range(n_lookback, len(y) - n_forecast + 1):
    #    X.append(y[i - n_lookback: i])
    #    Y.append(y[i: i + n_forecast])
    #
    #X = np.array(X)
    #Y = np.array(Y)
    #
    ## fit the model
    #model = Sequential()
    #model.add(LSTM(units=50, return_sequences=True, input_shape=(n_lookback, 1)))
    #model.add(LSTM(units=50))
    #model.add(Dense(n_forecast))
    #
    #model.compile(loss='mean_squared_error', optimizer='adam')
    #model.fit(X, Y, epochs=2, batch_size=32, verbose=0)
    #
    ## generate the forecasts
    #X_ = y[- n_lookback:]  # last available input sequence
    #X_ = X_.reshape(1, n_lookback, 1)
    #
    #Y_ = model.predict(X_).reshape(-1, 1)
    #Y_ = scaler.inverse_transform(Y_)
    #
    ## organize the results in a data frame
    #df_past = df_train[['Close']].reset_index()
    #df_past.rename(columns={'index': 'Date', 'Close': 'Actual'}, inplace=True)
    #df_past['Date'] = pd.to_datetime(df_past['Date'])
    #df_past['Forecast'] = np.nan
    #df_past['Forecast'].iloc[-1] = df_past['Actual'].iloc[-1]
    #
    #df_future = pd.DataFrame(columns=['Date', 'Actual', 'Forecast'])
    #df_future['Date'] = pd.date_range(start=df_past['Date'].iloc[-1] + pd.Timedelta(days=1), periods=n_forecast)
    #df_future['Forecast'] = Y_.flatten()
    #df_future['Actual'] = np.nan
    a,b = create_train_test_windows(x,y,n_lookback,n_forecast)
    bar.progress(20)
    stock_model = create_model(n_lookback,n_forecast,model_type='LSTM')
    bar.progress(40)
    stock_model.fit(a, b, epochs=2, batch_size=32)
    bar.progress(80)
    df_f,df_p = make_future_dataframe(data,x,y,n_lookback,n_forecast,stock_model,scaler)
    #m = Prophet()
    #m.fit(df_train)
    #future = m.make_future_dataframe(periods=period)
    #forecast = m.predict(future)
    st.subheader('Past data')
    st.write(df_p.tail())
    df_f.drop(columns=['Actual'],inplace=True)
    # Show and plot forecast
    st.subheader('Forecast data')
    st.write(df_f.tail())
    
    plot_forecast()
    
    bar.progress(100)
    #st.write(f'Forecast plot for {n_years} years')
    #fig1 = plot_plotly(m, forecast)
#st.plotly_chart(fig1)
#
#st.write("Forecast components")
#fig2 = m.plot_components(forecast)
#st.write(fig2)