import os 
import streamlit as st
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import date, timedelta

# --- Configuration ---
st.set_page_config(page_title="Stock Price Predictor", page_icon="📈", layout="wide")
plt.style.use('seaborn-v0_8-darkgrid')

# --- Helper Functions ---

# --- NEW FUNCTION TO SEARCH FOR TICKER SYMBOL ---
@st.cache_data(ttl=86400) # Cache the symbol lookup for 24 hours
def get_ticker_symbol(api_key, company_name):
    """Finds the best matching ticker symbol for a given company name."""
    url = f'https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords={company_name}&apikey={api_key}'
    try:
        r = requests.get(url)
        r.raise_for_status()
        data = r.json()
        if "bestMatches" in data and len(data["bestMatches"]) > 0:
            # Return the symbol of the best match
            return data["bestMatches"][0]["1. symbol"]
        else:
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Network error during ticker search: {e}")
        return None
    except ValueError:
        st.error("Error decoding JSON from API during ticker search.")
        return None

@st.cache_data(ttl=3600) # Cache data for 1 hour
def fetch_stock_data(api_key, stock_symbol):
    """Fetches historical stock data from Alpha Vantage."""
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={stock_symbol}&outputsize=full&apikey={api_key}'
    try:
        r = requests.get(url)
        r.raise_for_status()
        data = r.json()
        if "Time Series (Daily)" not in data:
            st.error(f"Error: Could not retrieve data for symbol '{stock_symbol}'. This might be a premium ticker or an API issue.")
            st.info(f"Server Response: {data.get('Note', 'No specific note from server.')}")
            return None
        
        df = pd.DataFrame(data['Time Series (Daily)']).T
        df.index = pd.to_datetime(df.index)
        df = df.sort_index(ascending=True)
        
        df = df.rename(columns={'1. open': 'Open', '2. high': 'High', '3. low': 'Low', '4. close': 'Close', '5. volume': 'Volume'})
        for col in df.columns:
            df[col] = pd.to_numeric(df[col])
            
        return df

    except requests.exceptions.RequestException as e:
        st.error(f"Network error: {e}")
        return None
    except ValueError:
        st.error("Error decoding JSON from API. The API might be down or the symbol is invalid.")
        return None

def preprocess_data(df):
    """Scales the 'Close' price and creates sequences for LSTM."""
    close_prices = df['Close'].values.reshape(-1, 1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(close_prices)
    
    time_step = 60
    X, y = [], []
    for i in range(len(scaled_data) - time_step - 1):
        X.append(scaled_data[i:(i + time_step), 0])
        y.append(scaled_data[i + time_step, 0])
        
    X, y = np.array(X), np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    
    return X, y, scaler, time_step

def build_and_train_model(X_train, y_train):
    """Builds and trains the LSTM model."""
    model = Sequential([
        LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        Dropout(0.2),
        LSTM(units=50, return_sequences=False),
        Dropout(0.2),
        Dense(units=25),
        Dense(units=1)
    ])
    
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, batch_size=32, epochs=50, verbose=0)
    
    return model

def make_prediction(model, scaler, last_60_days_scaled):
    """Makes a prediction for the next day."""
    X_test = np.array([last_60_days_scaled])
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    
    predicted_price_scaled = model.predict(X_test)
    predicted_price = scaler.inverse_transform(predicted_price_scaled)
    
    return predicted_price[0][0]
    
# --- Streamlit App UI ---

st.title("📈 Real-Time Stock Price Predictor")
st.markdown("This app uses an LSTM (Long Short-Term Memory) neural network to predict the next day's closing price of a stock.")

try:
    API_KEY = st.secrets["API_KEY"]
except KeyError:
    st.error("API_KEY not found in Streamlit secrets. Please add it to your `.streamlit/secrets.toml` file.")
    st.stop()

# --- MODIFIED USER INPUT ---
user_input = st.text_input("Enter Company Name or Ticker Symbol (e.g., Google, AAPL, Tesla):", "Google")

if st.button("Get Prediction"):
    if not user_input:
        st.warning("Please enter a company name or ticker symbol.")
    else:
        # --- MODIFIED LOGIC: First, find the ticker symbol ---
        stock_symbol = get_ticker_symbol(API_KEY, user_input)
        
        if stock_symbol is None:
            st.error(f"Could not find a matching stock ticker for '{user_input}'. Please try a more specific name.")
        else:
            st.info(f"Found ticker symbol: **{stock_symbol}**. Proceeding with prediction...")
            
            with st.spinner(f"Fetching data and training model for {stock_symbol}... This may take a moment."):
                # 1. Fetch Data
                df = fetch_stock_data(API_KEY, stock_symbol)
                
                if df is not None and not df.empty:
                    st.success(f"Successfully fetched data for {stock_symbol}. Latest data from {df.index[-1].date()}.")
                    
                    # 2. Preprocess Data
                    X_train, y_train, scaler, time_step = preprocess_data(df)
                    
                    # 3. Build and Train Model
                    model = build_and_train_model(X_train, y_train)
                    
                    # 4. Make Prediction
                    close_prices = df['Close'].values.reshape(-1, 1)
                    scaled_data = scaler.transform(close_prices)
                    last_60_days_scaled = scaled_data[-time_step:].flatten()
                    
                    predicted_price = make_prediction(model, scaler, last_60_days_scaled)
                    
                    # 5. Display Results
                    st.subheader(f"Prediction for {user_input} ({stock_symbol})")
                    last_close_price = df['Close'].iloc[-1]
                    st.metric(
                        label=f"Predicted Close Price for { (date.today() + timedelta(days=1)).strftime('%Y-%m-%d') }",
                        value=f"${predicted_price:,.2f}",
                        delta=f"${predicted_price - last_close_price:,.2f} ({((predicted_price - last_close_price) / last_close_price) * 100:.2f}%)"
                    )
                    
                    # 6. Plotting
                    st.subheader("Historical Data vs. Prediction")
                    fig, ax = plt.subplots(figsize=(16, 8))
                    
                    ax.plot(df.index[-365:], df['Close'][-365:], label='Historical Close Price', color='dodgerblue', linewidth=2)
                    
                    last_date = df.index[-1]
                    next_day = last_date + timedelta(days=1)
                    ax.plot([last_date, next_day], [df['Close'].iloc[-1], predicted_price], 'ro--', label='Predicted Price', markersize=8)
                    
                    ax.set_title(f'{user_input} ({stock_symbol}) Close Price History & Prediction', fontsize=20)
                    ax.set_xlabel('Date', fontsize=14)
                    ax.set_ylabel('Price (USD)', fontsize=14)
                    ax.legend(fontsize=12)
                    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    
                    st.pyplot(fig)

st.markdown("---")
st.warning(
    "**Disclaimer:** This is an educational tool and not financial advice. "
    "Stock market predictions are inherently uncertain. Please do your own research before making any investment decisions."
)