import streamlit as st
import datetime as dt
import pandas as pd
import yfinance as yf
import numpy as np
from scipy.stats import norm
from pandas_datareader import data as pdr
import plotly.offline as pyo
import plotly.graph_objs as go
from plotly.subplots import make_subplots

#######################
# Page configuration
st.set_page_config(
    page_title="Black Scholes Pricing Asset",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded")


# Custom CSS to inject into Streamlit
st.markdown("""
<style>
/* Adjust the size and alignment of the CALL and PUT value containers */
.metric-container {
    display: flex;
    justify-content: center;
    align-items: center;
    padding: 8px; /* Adjust the padding to control height */
    width: auto; /* Auto width for responsiveness, or set a fixed width if necessary */
    margin: 0 auto; /* Center the container */
    text-align: center;
}

/* Custom classes for CALL and PUT values */
.metric-call {
    background-color: #90ee90; /* Light green background */
    color: black; /* Black font color */
    margin-right: 10px; /* Spacing between CALL and PUT */
    border-radius: 10px; /* Rounded corners */
    text-align: center;
}

.metric-put {
    background-color: #ffcccb; /* Light red background */
    color: black; /* Black font color */
    border-radius: 10px; /* Rounded corners */
    text-align: centre;
}

/* Style for the value text */
.metric-value {
    font-size: 1.5rem; /* Adjust font size */
    font-weight: bold;
    margin: 0; /* Remove default margins */
    text-align: center;
}

/* Style for the label text */
.metric-label {
    font-size: 1rem; /* Adjust font size */
    margin-bottom: 4px; /* Spacing between label and value */
}

</style>
""", unsafe_allow_html=True)

# Sidebar for User Inputs
with st.sidebar:
    st.title("📊 Black-Scholes Model")
    st.write("Created by:")
    linkedin_url = "https://www.linkedin.com/in/prytm/"
    st.markdown(f'<a href="{linkedin_url}" target="_blank" style="text-decoration: none; color: inherit;"><img src="https://cdn-icons-png.flaticon.com/512/174/174857.png" width="25" height="25" style="vertical-align: middle; margin-right: 10px;">`Priya Tammam`</a>', unsafe_allow_html=True)

    stocks = st.text_input("Underlying Asset", value = "AAPL")
    strike = st.number_input("Strike Price", value=100.0)
    time_to_maturity = st.number_input("Time to Maturity (Years)", value=1.0)
    interest_rate = st.number_input("Risk-Free Interest Rate", value=0.05)

# (Include the BlackScholes class definition here)
def BlackScholes (S, K, T, r, sigma):
    d1 = (np.log(S/K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return call_price, put_price

# Stocks df
end = dt.datetime.now()
start = end - dt.timedelta(days=120)

df = yf.download(stocks, start, end)
df.columns = df.columns.get_level_values(0)

# Stocks Volatility
log_return = np.log(df['Close'] / df['Close'].shift(1))
log_return = log_return.dropna()
volatility = log_return.rolling(window=30).std() * np.sqrt(252)
bs_vol = volatility.iloc[-1]

# Current Price
current_price = df['Close'].iloc[-1]

# Candle Stick PLot
def plot_candlestick_volume(df, stocks):
    # Bollinger Bands Calculation
    df['MA10'] = df['Close'].rolling(window=10).mean()
    df['BB_upper'] = df['MA10'] + 2 * df['Close'].rolling(window=10).std()
    df['BB_lower'] = df['MA10'] - 2 * df['Close'].rolling(window=10).std()

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(f'{stocks} Price Chart', 'Volume'),
        row_width=[0.2, 0.7]
    )

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='OHLC'
    ), row=1, col=1)

    # Moving Averages
    fig.add_trace(go.Scatter(x=df.index, y=df['MA50'], line=dict(color='blue', width=1), name='MA50'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA200'], line=dict(color='grey', width=1), name='MA200'), row=1, col=1)

    # Bollinger Bands
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_upper'], line=dict(color='green', width=1), name='BB Upper'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['BB_lower'], line=dict(color='orange', width=1), name='BB Lower'), row=1, col=1)
    fig.add_trace(go.Scatter(x=df.index, y=df['MA10'], line=dict(color='purple', width=1), name='MA10'), row=1, col=1)

    # Volume
    fig.add_trace(go.Bar(x=df.index, y=df['Volume'], marker_color='red', showlegend=False), row=2, col=1)

    fig.update_layout(
        xaxis_tickfont_size=10,
        yaxis=dict(title='Price ($/Share)'),
        autosize=True,
        height=700,
        margin=dict(l=50, r=50, b=100, t=50, pad=4),
        paper_bgcolor='black',
        plot_bgcolor='black',
        font=dict(color='white')
    )

    fig.update(layout_xaxis_rangeslider_visible=False)
    return fig



# Main Page for Output Display
st.title("Black-Scholes Pricing Model")

# Table of Inputs
input_data = {
    "Current Asset Price": [current_price],
    "Strike Price": [strike],
    "Time to Maturity (Years)": [time_to_maturity],
    "Volatility (σ)": [bs_vol],
    "Risk-Free Interest Rate": [interest_rate],
}
input_df = pd.DataFrame(input_data)
st.table(input_df)

# Calculate Call and Put values
call_price, put_price = BlackScholes(strike, current_price, time_to_maturity, interest_rate, bs_vol)

# Display Call and Put Values in colored tables
col1, col2 = st.columns([1,1], gap="small")

with col1:
    # Using the custom class for CALL value
    st.markdown(f"""
        <div class="metric-container metric-call">
            <div>
                <div class="metric-label">Optimal CALL Value</div>
                <div class="metric-value">${call_price:.2f}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

with col2:
    # Using the custom class for PUT value
    st.markdown(f"""
        <div class="metric-container metric-put">
            <div>
                <div class="metric-label">Optimal PUT Value</div>
                <div class="metric-value">${put_price:.2f}</div>
            </div>
        </div>
    """, unsafe_allow_html=True)

st.markdown("")
st.title("Options Price - Interactive Candlestick Chart")
st.info("Explore how option prices fluctuate with varying 'Spot Prices and Volatility' levels using interactive heatmap parameters, all while maintaining a constant 'Strike Price'.")

# Interactive Sliders and Heatmaps for Call and Put Options
if not df.empty:
    df['MA50'] = df['Close'].rolling(window = 50, min_periods = 0).mean()
    df['MA200'] = df['Close'].rolling(window = 200, min_periods = 0).mean()

    # Call plot function
    st.subheader(f"{stocks} Candlestick Chart")
    candlestick_fig = plot_candlestick_volume(df, stocks)
    st.plotly_chart(candlestick_fig, use_container_width=True)
else:
    st.warning("⚠️ No data available for the selected ticker and date range.")
